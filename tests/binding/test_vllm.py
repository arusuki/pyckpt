import os
import time
import uuid
from io import BytesIO
from multiprocessing import Process
from typing import Type

from torch.multiprocessing import Queue
from transformers import AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from pyckpt import objects
from pyckpt.binding import torch as patch_torch
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

def _remove_kv_cache(core: EngineCore):
    ctx = core.vllm_config.compilation_config.static_forward_context
    for attn in ctx.values():
        del attn.kv_cache
    core.model_executor.driver_worker.worker.model_runner.kv_caches = None

def _remove_config(core: EngineCore):
    core.vllm_config.compilation_config.static_forward_context.clear()

def _remove_kv_cache_manager(core: EngineCore):
    scheduler = core.scheduler
    assert isinstance(scheduler, Scheduler)
    scheduler.kv_cache_manager = (
        type(scheduler.kv_cache_manager),
        {
            "kv_cache_config":scheduler.kv_cache_config,
            "max_model_len":scheduler.max_model_len,
            "enable_caching":scheduler.cache_config.enable_prefix_caching,
            "caching_hash_algo":scheduler.cache_config.prefix_caching_hash_algo,
            "use_eagle":scheduler.use_eagle,
            "log_stats":scheduler.log_stats,
            "enable_kv_cache_events":scheduler.enable_kv_cache_events,
        }
    )


def _build_engine_wo_scheduler(q: Queue):
    patch_torch.init()
    core = _make_engine_core()
    _remove_kv_cache(core)
    _remove_config(core)
    _remove_kv_cache_manager(core)

    core.model_executor = type(core.model_executor)
    file = BytesIO()
    _ = objects.dump(file, core)
    engine_data = file.getvalue()
    print(f"dump engine_data, size: {len(engine_data) / (1024 * 1024 * 1024)}GB")
    q.put(engine_data)

def _rebuild_engine_wo_scheduler(engine_data: bytes):
    file = BytesIO(engine_data)
    engine, _ = objects.load(file, {})
    assert isinstance(engine, EngineCore)
    assert isinstance(engine.model_executor, Type)

    engine.model_executor = engine.model_executor(engine.vllm_config)
    manager_cls, args = engine.scheduler.kv_cache_manager
    engine.scheduler.kv_cache_manager = manager_cls(**args)
    engine._initialize_kv_caches(engine.vllm_config)

    engine.add_request(make_request())
    assert len(engine.scheduler.waiting) == 1
    assert len(engine.scheduler.running) == 0

    _ = engine.step()
    assert len(engine.scheduler.waiting) == 0
    assert len(engine.scheduler.running) == 1



def test_vllm_rebuild_kv_cache():
    q = make_queue()
    dumper = run_spawned(_build_engine_wo_scheduler, q)
    engine_data = q.get()
    assert isinstance(engine_data, bytes)
    join_safe(dumper)

    reloader = run_spawned(_rebuild_engine_wo_scheduler, engine_data)
    join_safe(reloader)
