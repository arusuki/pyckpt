from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Type

from vllm.attention.layer import Attention
from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
from vllm.utils import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
import vllm.v1.engine.detokenizer as detokenizer
from tokenizers.decoders import DecodeStream
from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.worker.worker_base import WorkerWrapperBase

from pyckpt import objects
from itertools import chain

import torch


@contextmanager
def remove_kv_cache(core: EngineCore):
    ctx = core.vllm_config.compilation_config.static_forward_context
    attn_caches = [(attn, attn.kv_cache) for attn in ctx.values()]
    for attn in ctx.values():
        attn.kv_cache = [(cache.shape, cache.dtype, cache.device) for cache in attn.kv_cache]
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
        

@dataclass
class CapturedKVCacheManager:
    manager_cls: Type[KVCacheManager]
    args: dict
    cache_blocks: list[list]
    manager_states: tuple[list[dict[str, list[KVCacheBlock]]], dict[str, int]]
    cached_hash_to_block: dict[BlockHashWithGroupId, dict[int, KVCacheBlock]]
    req_to_block_hashes: defaultdict


@contextmanager
def remove_kv_cache_manager(core: EngineCore, cache_blocks: list[list]):
    scheduler = core.scheduler
    assert isinstance(scheduler, Scheduler)
    manager = scheduler.kv_cache_manager
    manager_states = [
        (type_manager.req_to_blocks, type_manager.num_cached_block)
        for type_manager in manager.coordinator.single_type_managers
    ]

    caching_hash_algo = "sha256" if manager.caching_hash_fn is sha256 else "builtin"
    manager_args = {
        "kv_cache_config":         manager.kv_cache_config,
        "max_model_len":           manager.max_model_len,
        "enable_caching":          manager.enable_caching,
        "caching_hash_algo":       caching_hash_algo,
        "use_eagle":               manager.use_eagle,
        "log_stats":               manager.log_stats,
        "enable_kv_cache_events":  manager.block_pool.enable_kv_cache_events,
    }
    scheduler.kv_cache_manager = CapturedKVCacheManager(
        manager_cls=type(manager),
        args = manager_args,
        cache_blocks = cache_blocks,
        manager_states = manager_states,
        cached_hash_to_block = manager.block_pool.cached_block_hash_to_block,
        req_to_block_hashes = manager.req_to_block_hashes,
    )

    try:
        yield
    finally:
        scheduler.kv_cache_manager = manager

@contextmanager
def remove_model_executor_worker(core: EngineCore):
    executor = core.model_executor
    assert isinstance(executor, UniProcExecutor), f"invliad executor type: {type(executor)}"
    driver_worker = executor.driver_worker
    executor.driver_worker = None
    rpc_rank = driver_worker.rpc_rank
    worker = driver_worker.worker
    assert isinstance(worker, Worker)
    core.model_executor = executor, (worker, rpc_rank)
    try:
        yield
    finally:
        core.model_executor = executor
        executor.driver_worker = driver_worker

@contextmanager
def remove_model_executor(core: EngineCore):
    executor = core.model_executor
    assert isinstance(executor, UniProcExecutor)
    model_runner = executor.driver_worker.worker.model_runner
    assert isinstance(model_runner, GPUModelRunner)
    core.model_executor = type(executor), model_runner.requests
    try:
        yield
    finally:
        core.model_executor = executor

@contextmanager
def prepare_engine(core: EngineCore):
    cache_blocks = get_cache_blocks(core)
    with ExitStack() as stack:
        stack.enter_context(remove_forward_config(core))
        stack.enter_context(remove_kv_cache(core))
        stack.enter_context(remove_kv_cache_manager(core, cache_blocks))
        stack.enter_context(remove_model_executor(core))
        yield

def collect_attention_blocks(ctx: dict[str, any], manager_blocks: tuple[list[KVCacheBlock]]):
    if not all(
        len(attn.kv_cache) == 1 for attn in ctx.values()
    ):
        raise NotImplementedError("attention with multiple cache tensors")
    cache_blocks = {}
    for blocks in manager_blocks:
        for block in blocks:
            c = {}
            for layer_name, attn in ctx.items():
                cache_tensor = attn.kv_cache[0]
                c[layer_name] = \
                (cache_tensor[0][block.block_id].to("cpu"),\
                 cache_tensor[1][block.block_id].to("cpu"))
            cache_blocks[block.block_id] = c
    return cache_blocks


def get_cache_tensors(core: EngineCore) -> list[list[torch.Tensor]]:
    """
    return: [group_id, layer_id] -> cache_tensor
    """
    assert isinstance(core.scheduler, Scheduler)
    kv_cache_config = core.scheduler.kv_cache_config
    layers = get_layers_from_vllm_config(core.vllm_config, Attention)
    assert len(core.vllm_config.compilation_config.static_forward_context) \
       ==  len(layers)
    cache_tensors = []
    for group_spec in kv_cache_config.kv_cache_groups:
        attns = [layers[name] for name in group_spec.layer_names]
        if any(len(attn.kv_cache) > 1 for attn in attns):
            raise NotImplementedError("pipeline parallelism")
        cache_tensors.append(list(attn.kv_cache[0] for attn in attns))
    return cache_tensors

CacheBlocks = list[list[tuple[list[int], torch.Tensor, torch.Tensor]]]

def get_cache_blocks(core: EngineCore) -> CacheBlocks:
    """
    return: [group_id, layer_id] -> (block_indices, k_cache_blocks, v_cahce_blocks)
    """
    cache_blocks = {}
    scheduler = core.scheduler
    assert isinstance(scheduler, Scheduler)
    cache_tensors = get_cache_tensors(core)
    cache_blocks = [ [] for _ in cache_tensors ]
    requests: dict[str, CachedRequestState] = \
      core.model_executor.driver_worker.worker.model_runner.requests
    for req in requests.values():
        block_groups = req.block_ids
        assert len(block_groups) == len(cache_tensors)
        for cache, blocks, cache_tensor in \
          zip(cache_blocks, block_groups, cache_tensors):
            print(f"saved blocks: {blocks}")
            block_indices = blocks.copy()
            cache.extend(
                (block_indices, layer[0][block_indices], layer[1][block_indices])
                for layer in cache_tensor
            )
    return cache_blocks


def set_cache_blocks(core: EngineCore, cache_blocks: CacheBlocks):
    scheduler = core.scheduler
    assert isinstance(scheduler, Scheduler)
    cache_tensors = get_cache_tensors(core)

    if len(cache_blocks) > 1:
        raise NotImplementedError("pipeline parallelism")
    for layer_tensors, layer_cache_blocks in zip(cache_tensors, cache_blocks):
        for layer_tensor, (block_indices, k_cache, v_cache) in \
          zip(layer_tensors, layer_cache_blocks):
            layer_tensor[0][block_indices] = k_cache
            layer_tensor[1][block_indices] = v_cache

def rebuild_core_executor_worker(core: EngineCore):
    executor, (worker, rpc_rank) = core.model_executor
    assert isinstance(executor, UniProcExecutor)
    assert isinstance(worker, Worker)
    with set_current_vllm_config(core.vllm_config):
        init_worker_distributed_environment(worker.vllm_config, worker.rank,
                                            worker.distributed_init_method,
                                            worker.local_rank)
    executor.driver_worker = WorkerWrapperBase(core.vllm_config, rpc_rank)
    executor.driver_worker.worker = worker
    core.model_executor = executor

    for attn in get_layers_from_vllm_config(core.vllm_config, Attention).values():
        attn.kv_cache = [
            torch.zeros(shape, dtype=dtype, device=device) \
              for (shape, dtype, device) in attn.kv_cache]

def rebuild_core_executor(core: EngineCore):
    executor_cls, requests = core.model_executor
    core.model_executor = executor_cls(core.vllm_config)

    model_runner = core.model_executor.driver_worker.worker.model_runner
    assert isinstance(model_runner, GPUModelRunner)
    model_runner.requests = requests
    core._initialize_kv_caches(core.vllm_config)

def rebuild_core_kv_cache_manager(core: EngineCore):
    captured = core.scheduler.kv_cache_manager
    assert isinstance(captured, CapturedKVCacheManager)
    manager = captured.manager_cls(**captured.args)

    s_managers = manager.coordinator.single_type_managers
    allocated_blocks: dict[int, KVCacheBlock] = {}
    block_pool = manager.block_pool
    num_groups = len(s_managers)
    assert len(captured.manager_states) == num_groups
    for s_manager, (req_to_block, num_cached_block) in \
      zip(s_managers, captured.manager_states):
        assert len(s_manager.req_to_blocks) == 0
        s_manager.req_to_blocks = req_to_block
        s_manager.num_cached_block = num_cached_block
        for blocks in req_to_block.values():
            allocated_blocks.update((block.block_id, block) for block in blocks)

    for block_id, block in allocated_blocks.items():
        old_block = manager.block_pool.blocks[block_id]
        block_pool.free_block_queue.remove(old_block)
        manager.block_pool.blocks[block_id] = block
    
    block_pool.cached_block_hash_to_block = captured.cached_hash_to_block
    manager.req_to_block_hashes = captured.req_to_block_hashes
    core.scheduler.kv_cache_manager = manager

    set_cache_blocks(core, captured.cache_blocks)

def rebuild_engine(dumped_core: bytes, objs: dict):
    untyped_stores = objs["storage"]
    assert isinstance(untyped_stores, bytes)
    objs["storage"] = objects.load_untyped_storages(BytesIO(untyped_stores))
    core, _ = objects.load(BytesIO(dumped_core), objs)
    assert isinstance(core, EngineCore)
    rebuild_core_executor(core)
    rebuild_core_kv_cache_manager(core)
    return core

def reduce_engine_core(core: EngineCore):
    with prepare_engine(core):
        dumped = BytesIO()
        storages = objects.dump(dumped, core)
        untyped_stores = storages["storage"]
        store_data = BytesIO()
        objects.save_untyped_storages(store_data, untyped_stores)
        storages["storage"] = store_data.getvalue()
        return (rebuild_engine, (dumped.getvalue(), storages))

decode_streams: dict[int, bool] = {}

def create_decode_stream(skip_special_tokens: bool):
    stream = DecodeStream(skip_special_tokens)
    decode_streams[id(stream)] = skip_special_tokens
    return stream

def reduce_decode_stream(stream: DecodeStream):
    assert id(stream) in decode_streams
    return create_decode_stream, (decode_streams[id(stream)],)

def init():
    detokenizer.DecodeStream = create_decode_stream
 
