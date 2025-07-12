from contextlib import ExitStack, contextmanager
from io import BytesIO

from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.worker.gpu_worker import Worker

from pyckpt import objects


@contextmanager
def remove_kv_cache(core: EngineCore):
    ctx = core.vllm_config.compilation_config.static_forward_context
    attn_caches = [(attn, attn.kv_cache) for attn in ctx.values()]
    for attn in ctx.values():
        del attn.kv_cache
    model_executor = core.model_executor
    assert isinstance(model_executor, UniProcExecutor), f"invliad executor: {model_executor}"
    worker = model_executor.driver_worker.worker
    assert isinstance(worker, Worker)
    worker_cache = worker.model_runner.kv_caches
    worker.model_runner.kv_caches = None
    try:
        yield
    finally:
        for attn, kv_cache in attn_caches:
            attn.kv_cache = kv_cache
        worker.model_runner.kv_caches = worker_cache

@contextmanager
def remove_forward_config(core: EngineCore):
    config = core.vllm_config.compilation_config
    ctx = config.static_forward_context
    config.static_forward_context = {}
    try:
        yield
    finally:
        config.static_forward_context = ctx
        

@contextmanager
def remove_kv_cache_manager(core: EngineCore):
    scheduler = core.scheduler
    assert isinstance(scheduler, Scheduler)
    manager = scheduler.kv_cache_manager
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
    try:
        yield
    finally:
        scheduler.kv_cache_manager = manager

@contextmanager
def remove_model_executor(core: EngineCore):
    executor = core.model_executor
    assert isinstance(executor, UniProcExecutor), f"invliad executor type: {type(executor)}"
    core.model_executor = type(core.model_executor)
    try:
        yield
    finally:
        core.model_executor = executor

@contextmanager
def prepare_engine(core: EngineCore):
    with ExitStack() as stack:
        stack.enter_context(remove_forward_config(core))
        stack.enter_context(remove_kv_cache(core))
        stack.enter_context(remove_kv_cache_manager(core))
        stack.enter_context(remove_model_executor(core))
        yield

def rebuild_engine(dumped_core: EngineCore):
    core, _ = objects.load(BytesIO(dumped_core), {})
    assert isinstance(core, EngineCore)
    core.model_executor = core.model_executor(core.vllm_config)
    manager_cls, args = core.scheduler.kv_cache_manager
    core.scheduler.kv_cache_manager = manager_cls(**args)
    core._initialize_kv_caches(core.vllm_config)
    return core

def reduce_engine_core(core: EngineCore):
    with prepare_engine(core):
        dumped = BytesIO()
        objects.dump(dumped, core)
        return (rebuild_engine, (dumped.getvalue(),))
