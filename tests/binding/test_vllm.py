import os
import time
import uuid
from functools import partial
from io import BytesIO, StringIO
from multiprocessing import Process
from typing import Any, Callable, Optional

import cloudpickle
import ray
import torch
from dill import Unpickler
from tokenizers.decoders import DecodeStream
from torch.multiprocessing import Queue
from transformers import AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.worker.worker_base import WorkerWrapperBase

from pyckpt import objects
from pyckpt.binding import torch as patch_torch
from pyckpt.binding import vllm as patch_vllm
from pyckpt.binding.vllm import (
    CapturedGPUModelRunner,
    get_cache_blocks_v1,
    get_req_cache_block_ids,
    prepare_engine,
    reduce_decode_stream,
    reduce_engine_core,
)
from pyckpt.objects import Pickler
from tests.utils import (
    make_queue,
    restore_random_states,
    run_spawned,
    save_random_states,
)

# MODEL_NAME =  os.path.expanduser("~/testp/Qwen2.5-7B-Instruct-GPTQ-Int8")
MODEL_NAME = "/docker/data/HF_MODELS/Qwen2.5-7B-Instruct-GPTQ-Int8"
# MODEL_NAME =  "/home/yuuka/testp/opt-125m"
PROMPT = "implement quick sort in C programming language: "


def make_request() -> EngineCoreRequest:
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids
    print(f"prompt tokens: {len(PROMPT_TOKENS)}")
    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt_token_ids=PROMPT_TOKENS,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(
            output_kind=RequestOutputKind.DELTA, max_tokens=2048
        ),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def join_safe(process: Process):
    process.join()
    assert process.exitcode == 0


def _make_engine_core(tp_size=1, pp_size=1):
    os.environ["VLLM_USE_V1"] = "1"
    engine_args = EngineArgs(
        model=MODEL_NAME,
        compilation_config=0,
        enforce_eager=True,
        max_model_len=8192,
        # gpu_memory_utilization=0.3,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size
    )
    vllm_config = engine_args.create_engine_config()
    assert vllm_config.compilation_config.level == 0
    executor_class = Executor.get_class(vllm_config)
    engine_core = EngineCore(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
    )
    return engine_core


def _step_engine(core: EngineCore):
    core.add_request(make_request())
    assert len(core.scheduler.waiting) == 1
    assert len(core.scheduler.running) == 0

    _ = core.step()
    assert len(core.scheduler.waiting) == 0
    assert len(core.scheduler.running) == 1


def _prepare_and_step_engine():
    core = _make_engine_core()
    with prepare_engine(core):
        pass
    _step_engine(core)


def _test_reduce_engine(
    q: Queue,
    test_func: Optional[Callable[[EngineCore], Any]] = None,
    tp_size=1,
):
    try:
        patch_torch.init()
        core = _make_engine_core(tp_size)
        ret = None
        if test_func:
            ret = test_func(core)
        file = BytesIO()
        pickler = Pickler(file)
        pickler.dispatch_table[EngineCore] = reduce_engine_core
        pickler.dispatch_table[DecodeStream] = reduce_decode_stream
        pickler.dump(ret)
        pickler.dump(core)
        storages = pickler.consume_persisted()
        engine_data = file.getvalue()
        q.put((engine_data, storages))
    finally:
        core.shutdown()


def _test_rebuild_engine(
    engine_data: bytes,
    storages: dict,
    test_func: Callable[[EngineCore], None],
    q: Optional[Queue] = None,
):
    core: Optional[EngineCore] = None
    try:
        unpickler = objects.Unpickler(BytesIO(engine_data), storages)
        args = unpickler.load()
        core = unpickler.load()
        if args:
            ret = test_func(core, args)
        else:
            ret = test_func(core)

        if q:
            file = BytesIO()
            pickler = Pickler(file)
            pickler.dispatch_table[EngineCore] = reduce_engine_core
            pickler.dispatch_table[DecodeStream] = reduce_decode_stream
            pickler.dump(ret)
            q.put(file.getvalue())
    finally:
        if core:
            core.shutdown()


def test_vllm_prepare_engine():
    ut = run_spawned(_prepare_and_step_engine)
    join_safe(ut)


def test_vllm_reduce_engine():
    q = make_queue()
    dumper = run_spawned(_test_reduce_engine, q)
    engine_data, storages = q.get()
    assert isinstance(engine_data, bytes)
    join_safe(dumper)

    reloader = run_spawned(_test_rebuild_engine, engine_data, storages, _step_engine)
    join_safe(reloader)


def _step_engine_and_process(
    num_step: int,
    core: EngineCore,
    args: Optional[tuple[OutputProcessor, dict, StringIO]] = None,
):
    patch_vllm.init()
    rng_states = None

    if not args:
        output = StringIO()
        request = make_request()
        core.add_request(request)
        tokenizer = TokenizerGroup(
            tokenizer_id=MODEL_NAME,
            enable_lora=False,
            max_num_seqs=256,
            max_input_length=None,
        )
        processor = OutputProcessor(tokenizer, False)
        processor.add_request(request, PROMPT)
    else:
        processor, rng_states, output = args

    if rng_states:
        restore_random_states(rng_states)

    for _ in range(num_step):
        outs = core.step()[0].get(0)
        assert outs.outputs is not None
        # print(outs.outputs)
        processed_outputs = processor.process_outputs(outs.outputs)
        assert len(processed_outputs.request_outputs) == 1
        request_output = processed_outputs.request_outputs[0]
        output.write(request_output.outputs[0].text)

    print_req_blocks(core)

    return processor, save_random_states(), output


def print_req_blocks(core: EngineCore):
    # scheduler = core.scheduler
    # assert isinstance(scheduler, Scheduler)
    # executor = core.model_executor
    # assert isinstance(executor, UniProcExecutor)
    # model_runner = executor.driver_worker.worker.model_runner
    # assert isinstance(model_runner, GPUModelRunner)
    # cached_req_states = model_runner.requests
    #
    # for req, cached_req in zip(scheduler.requests, cached_req_states.values()):
    #     print(
    #         "cache blocks(manager): ", scheduler.kv_cache_manager.get_block_ids(req)
    #     )
    #     print(
    #         "cache blocks(model_runner): ", cached_req.block_ids
    #     )
    pass
def print_cache_blocks(cache_blocks: list[list[tuple[int, torch.Tensor]]]):
    assert len(cache_blocks) == 1
    blocks = cache_blocks[0]
    print(f"block_ids: {[block[0] for block in blocks]}")


def test_reduce_output_processor():
    patch_vllm.init()
    request = make_request()
    tokenizer = TokenizerGroup(
        tokenizer_id=MODEL_NAME,
        enable_lora=False,
        max_num_seqs=256,
        max_input_length=None,
    )
    processor = OutputProcessor(tokenizer, False)
    processor.add_request(request, PROMPT)
    file = BytesIO()
    pickler = Pickler(file)
    pickler.dispatch_table[DecodeStream] = reduce_decode_stream
    pickler.dump(processor)
    file.seek(0)
    processor = Unpickler(file).load()
    assert isinstance(processor, OutputProcessor)


def test_vllm_engine_step_after_dump():
    def get_text_from_data(engine_data: bytes):
        pickler = Unpickler(BytesIO(engine_data))
        ret = pickler.load()
        assert isinstance(ret, tuple) and len(ret) == 3
        output = ret[2]
        assert isinstance(output, StringIO)
        return output.getvalue()

    def step(num_step: int):
        return partial(_step_engine_and_process, num_step)

    q = make_queue()
    dumper = run_spawned(_test_reduce_engine, q, step(128))
    engine_data, _ = q.get()
    assert isinstance(engine_data, bytes)
    join_safe(dumper)

    reference = get_text_from_data(engine_data)
    print(reference)

    dumper = run_spawned(_test_reduce_engine, q, step(64))
    engine_data, storages = q.get()
    assert isinstance(engine_data, bytes)
    join_safe(dumper)

    reloader = run_spawned(_test_rebuild_engine, engine_data, storages, step(64), q)
    engine_data = q.get()
    join_safe(reloader)

    interrupted = get_text_from_data(engine_data)
    print(interrupted)

    assert reference == interrupted


def test_vllm_reduce_engine_tp():
    def get_text_from_data(engine_data: bytes):
        pickler = Unpickler(BytesIO(engine_data))
        ret = pickler.load()
        assert isinstance(ret, tuple) and len(ret) == 3
        output = ret[2]
        assert isinstance(output, StringIO)
        return output.getvalue()

    def step(num_step: int):
        return partial(_step_engine_and_process, num_step)

    q = make_queue()
    dumper = run_spawned(_test_reduce_engine, q, step(64), 2)
    engine_data, storages = q.get()
    join_safe(dumper)

    reference = get_text_from_data(engine_data)
    print(reference)

    reloader = run_spawned(_test_rebuild_engine, engine_data, storages, step(64), q)
    engine_data = q.get()
    join_safe(reloader)

    interrupted = get_text_from_data(engine_data)
    print(interrupted)

def _test_pp():
    core = _make_engine_core(2, 2)
    print("executor type: ", type(core.model_executor))
    assert isinstance(core.model_executor, RayDistributedExecutor)
    kv_cache_config = core.scheduler.kv_cache_config
    block_ids = get_req_cache_block_ids(core)

    def capture_ray_distributed_executor(worker: WorkerWrapperBase):
        cache_blocks = get_cache_blocks_v1(
            worker.vllm_config,
            block_ids,
            kv_cache_config,
        ) 
        return [
            CapturedGPUModelRunner(
                worker.worker.model_runner.requests,
                cache_blocks,
                save_random_states(),
            )
        ]

    ret = []
    for worker in core.model_executor.workers:
        ret.append(
            worker.execute_method.remote(
                cloudpickle.dumps(capture_ray_distributed_executor)
            )
        )
    ret = ray.get(ret)
    print(type(ret))


def test_vllm_reduce_engine_pp():
    dumper = run_spawned(_test_pp)
    join_safe(dumper)

