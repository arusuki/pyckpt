import os
import time
import uuid
from io import BytesIO
from multiprocessing import Process

from torch.multiprocessing import Queue
from transformers import AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from pyckpt import objects
from pyckpt.binding import torch as patch_torch
from pyckpt.binding.vllm import prepare_engine, reduce_engine_core
from pyckpt.objects import Pickler
from tests.utils import make_queue, run_spawned

MODEL_NAME =  "/home/yuuka/testp/opt-125m"
PROMPT = "Hello my name is Robert and I love quantization kernels"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids

def make_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt_token_ids=PROMPT_TOKENS,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

def join_safe(process: Process):
    process.join()
    assert process.exitcode == 0

def _make_engine_core():
    os.environ["VLLM_USE_V1"] = "1"
    engine_args = EngineArgs(model=MODEL_NAME, compilation_config=0, enforce_eager=True)
    vllm_config = engine_args.create_engine_config()

    print("static_forward_context: ", vllm_config.compilation_config.static_forward_context)

    assert vllm_config.compilation_config.level == 0

    executor_class = Executor.get_class(vllm_config)
    engine_core = EngineCore(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=True,
    )
    return engine_core


def _prepare_and_step_engine():
    core = _make_engine_core()
    with prepare_engine(core):
        pass

    core.add_request(make_request())
    assert len(core.scheduler.waiting) == 1
    assert len(core.scheduler.running) == 0

    _ = core.step()
    assert len(core.scheduler.waiting) == 0
    assert len(core.scheduler.running) == 1

def test_vllm_prepare_engine():
    ut = run_spawned(_prepare_and_step_engine)
    join_safe(ut)

def _test_reduce_engine(q: Queue):
    patch_torch.init()
    core = _make_engine_core()
    print(f"ext1: {core.model_executor}")
    file = BytesIO()
    pickler = Pickler(file)
    pickler.dispatch_table[EngineCore] = reduce_engine_core
    _ = pickler.dump(core)
    engine_data = file.getvalue()
    print(f"dump engine_data, size: {len(engine_data) / (1024 * 1024 * 1024)}GB")
    q.put(engine_data)

def _test_rebuild_engine(engine_data: bytes):
    core, _ = objects.load(BytesIO(engine_data), {})

    core.add_request(make_request())
    assert len(core.scheduler.waiting) == 1
    assert len(core.scheduler.running) == 0

    _ = core.step()
    assert len(core.scheduler.waiting) == 0
    assert len(core.scheduler.running) == 1

def test_vllm_reduce_engine():
    q = make_queue()
    dumper = run_spawned(_test_reduce_engine, q)
    engine_data = q.get()
    assert isinstance(engine_data, bytes)
    join_safe(dumper)

    reloader = run_spawned(_test_rebuild_engine, engine_data)
    join_safe(reloader)

