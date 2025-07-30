import random
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Type

import dill
import numpy as np
import torch
import vllm.v1.engine.detokenizer as detokenizer
from tokenizers.decoders import DecodeStream
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.utils import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.worker.worker_base import WorkerWrapperBase

from pyckpt import objects


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
    manager_states: tuple[list[dict[str, list[KVCacheBlock]]], dict[str, int]]
    cached_hash_to_block: dict[BlockHashWithGroupId, dict[int, KVCacheBlock]]
    req_to_block_hashes: defaultdict


@contextmanager
def remove_kv_cache_manager(core: EngineCore):
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
        manager_states = manager_states,
        cached_hash_to_block = manager.block_pool.cached_block_hash_to_block,
        req_to_block_hashes = manager.req_to_block_hashes,
    )

    try:
        yield
    finally:
        scheduler.kv_cache_manager = manager


def save_random_states() -> bytes:
    states = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    return dill.dumps(states)

def restore_random_states(states_data: bytes):
    states = dill.loads(states_data)
    random.setstate(states['python_random'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    if torch.cuda.is_available() and states['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(states['torch_cuda'])

@dataclass
class CapturedGPUModelRunner:
    requests: dict[str, CachedRequestState]
    cache_blocks: "CacheBlocks"
    random_states: bytes

def collect_worker_cache_blocks(
    worker: WorkerWrapperBase,
    block_ids: list[list[int]],
    kv_cache_config: KVCacheConfig,
):
    gpu_worker = worker.worker
    if not isinstance(gpu_worker, GPUWorker):
        raise ValueError(f"unsupported worker type: {type(gpu_worker)}")
    cache_blocks = \
      get_cache_blocks_v1(worker.vllm_config, block_ids, kv_cache_config)
    return CapturedGPUModelRunner(
        requests = gpu_worker.model_runner.requests,
        cache_blocks = cache_blocks,
        random_states = save_random_states()
    )

@contextmanager
def remove_model_executor(core: EngineCore):
    block_ids = get_req_cache_block_ids(core)
    kv_cache_config = core.scheduler.kv_cache_config

    def capture_uni_proc_executor(executor: UniProcExecutor):
        model_runner = executor.driver_worker.worker.model_runner
        assert isinstance(model_runner, GPUModelRunner)
        cache_blocks = get_cache_blocks_v1(
            core.vllm_config,
            block_ids,
            kv_cache_config,
        )
        return [
            CapturedGPUModelRunner(
                model_runner.requests,
                cache_blocks,
                save_random_states(),
            )
        ]

    def capture_multi_proc_executor(
        executor: MultiprocExecutor,
    ) -> list[CapturedGPUModelRunner]:
        return executor.collective_rpc(
            collect_worker_cache_blocks, args=(block_ids,kv_cache_config)
        )

    def capture_ray_distributed_executor(
        executor: RayDistributedExecutor
    ) -> list[CapturedGPUModelRunner]:
        raise NotImplementedError("capture_ray_distributed_executor")

    executor = core.model_executor
    if isinstance(executor, UniProcExecutor):
        runners = capture_uni_proc_executor(executor)
    elif isinstance(executor, MultiprocExecutor):
        runners = capture_multi_proc_executor(executor)
    elif isinstance(executor, RayDistributedExecutor):
        runners = capture_ray_distributed_executor(executor)
    else:
        raise ValueError(f"unsupported executor: {executor}")

    executor = core.model_executor
    core.model_executor = type(executor), runners
    try:
        yield
    finally:
        core.model_executor = executor

@contextmanager
def prepare_engine(core: EngineCore):
    with ExitStack() as stack:
        stack.enter_context(remove_model_executor(core))
        stack.enter_context(remove_kv_cache_manager(core))
        stack.enter_context(remove_forward_config(core))
        yield

def get_cache_tensors_v1(vllm_config: VllmConfig):
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    assert all(len(attn.kv_cache) == 1 for attn in layers.values())
    return {
        layer_name: attn.kv_cache[0] for layer_name, attn in layers.items()
    }

CacheBlocks = dict[str, tuple[list[int], torch.Tensor, torch.Tensor]]

def get_cache_blocks_v1(
    vllm_config: VllmConfig,
    block_ids: list[list[int]],
    kv_cache_config: KVCacheConfig,
    device: Optional[torch.device | str] = None,
) -> CacheBlocks:
    """
    return: {[layer_name]: (block_indices, k_cache_blocks, v_cahce_blocks)}
    """
    if device is None:
       device = torch.accelerator.current_accelerator()
    cache_tensors = get_cache_tensors_v1(vllm_config)
    group_specs = kv_cache_config.kv_cache_groups
    assert len(block_ids) == len(group_specs)
    cache_blocks = {}
    for group_spec, blocks in zip(group_specs, block_ids):
        for layer_name in group_spec.layer_names:
            if layer_name not in cache_tensors:
                continue
            cache_tensor = cache_tensors[layer_name]
            cache_blocks[layer_name] = (
                blocks, 
                cache_tensor[0][blocks].to(device).share_memory_(),
                cache_tensor[1][blocks].to(device).share_memory_(),
            )
    return cache_blocks

def set_cache_blocks_v1(vllm_config: VllmConfig, cache_blocks: CacheBlocks):
    cache_tensors = get_cache_tensors_v1(vllm_config)
    for layer_name, cache_tensor in cache_tensors.items():
        # TODO(arusuki): optimize cache block copy here
        blocks, k_cache, v_cache = cache_blocks[layer_name]
        cache_tensor[0][blocks] = k_cache.to(cache_tensor.device)
        cache_tensor[1][blocks] = v_cache.to(cache_tensor.device)


def get_req_cache_block_ids(core: EngineCore) -> list[list[int]]:
    scheduler = core.scheduler
    assert isinstance(scheduler, Scheduler)
    manager = scheduler.kv_cache_manager
    num_groups = len(manager.kv_cache_config.kv_cache_groups)
    cache_blocks = [[] for _ in range(num_groups)]
    for req in scheduler.requests:
        block_groups =  manager.get_block_ids(req)
        for group_ids, block_ids in zip(cache_blocks, block_groups):
            group_ids.extend(block_ids)
    return cache_blocks

def set_worker_kv_cache(
    worker: WorkerWrapperBase,
    runners: list[CapturedGPUModelRunner],
):
    gpu_worker = worker.worker
    if not isinstance(gpu_worker, GPUWorker):
        raise ValueError(f"unsupported worker type: {type(gpu_worker)}")
    print(f"set worker at rank {worker.rpc_rank}")
    captured = runners[worker.rpc_rank]
    gpu_worker.model_runner.requests = captured.requests
    set_cache_blocks_v1(worker.vllm_config, captured.cache_blocks)
    restore_random_states(captured.random_states)

def rebuild_core_executor(core: EngineCore):
    def restore_uni_proc_executor(
        executor: UniProcExecutor,
        runners: list[CapturedGPUModelRunner],
    ):
        model_runner = executor.driver_worker.worker.model_runner
        assert isinstance(model_runner, GPUModelRunner)
        assert len(runners) == 1
        model_runner.requests = runners[0].requests
        set_cache_blocks_v1(core.vllm_config, runners[0].cache_blocks)

    def restore_multi_proc_executor(
        executor: MultiprocExecutor,
        runners: list[CapturedGPUModelRunner],
    ):
        return executor.collective_rpc(
            set_worker_kv_cache,
            args=(runners,)
        )

    runners: list[CapturedGPUModelRunner]
    executor_cls, runners = core.model_executor
    executor = executor_cls(core.vllm_config)
    core.model_executor = executor
    core._initialize_kv_caches(core.vllm_config)
    if isinstance(executor, UniProcExecutor):
        restore_uni_proc_executor(core.model_executor, runners)
    elif isinstance(executor, MultiprocExecutor):
        restore_multi_proc_executor(executor, runners)
    else:
        raise NotImplementedError(f"unsupported executor: {type(executor)}")

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

def rebuild_engine(dumped_core: bytes, objs: dict):
    untyped_stores = objs["storage"]
    assert isinstance(untyped_stores, bytes)
    objs["storage"] = objects.load_untyped_storages(BytesIO(untyped_stores))
    core, _ = objects.load(BytesIO(dumped_core), objs)
    assert isinstance(core, EngineCore)
    rebuild_core_kv_cache_manager(core)
    rebuild_core_executor(core)
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
 
