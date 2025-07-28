from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Type

import torch
import vllm.v1.engine.detokenizer as detokenizer
from tokenizers.decoders import DecodeStream
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.utils import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
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

@dataclass
class CapturedGPUModelRunner:
    requests: dict[str, CachedRequestState]
    cache_blocks: "CacheBlocks"

def collect_worker_cache_blocks(
    worker: WorkerWrapperBase,
    layer_groups: list[list[str]],
    block_ids: list[list[int]],
):
    gpu_worker = worker.worker
    if not isinstance(gpu_worker, GPUWorker):
        raise ValueError(f"unsupported worker type: {type(gpu_worker)}")
    cache_blocks = \
      get_cache_blocks_v1(worker.vllm_config, layer_groups, block_ids)
    return CapturedGPUModelRunner(
        requests = gpu_worker.model_runner.requests,
        cache_blocks = cache_blocks,
    )

@contextmanager
def remove_model_executor(core: EngineCore):
    layer_groups = get_layer_groups(core)
    block_ids = get_req_cache_block_ids(core)

    def capture_uni_proc_executor(executor: UniProcExecutor):
        model_runner = executor.driver_worker.worker.model_runner
        assert isinstance(model_runner, GPUModelRunner)
        cache_blocks = get_cache_blocks_v1(
            core.vllm_config,
            layer_groups, 
            block_ids,
        )
        assert len(cache_blocks) == 1
        return [CapturedGPUModelRunner(model_runner.requests, cache_blocks)]

    def capture_multi_proc_executor(
        executor: MultiprocExecutor,
    ) -> list[CapturedGPUModelRunner]:
        return executor.collective_rpc(
            collect_worker_cache_blocks, args=(layer_groups, block_ids)
        )

    executor = core.model_executor
    if isinstance(executor, UniProcExecutor):
        runners = capture_uni_proc_executor(executor)
    elif isinstance(executor, MultiprocExecutor):
        runners = capture_multi_proc_executor(executor)
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

def get_cache_tensors_v1(
    vllm_config: VllmConfig,
    layer_groups: list[list[str]],
):
    cache_tensors: list[list[torch.Tensor]] = []
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer_names in layer_groups:
        attns = [layers[name] for name in layer_names]
        if any(len(attn.kv_cache) > 1 for attn in attns):
            raise NotImplementedError("pipeline parallelism")
        cache_tensors.append(list(attn.kv_cache[0] for attn in attns))
    return cache_tensors

CacheBlocks = list[list[tuple[list[int], torch.Tensor, torch.Tensor]]]

def get_cache_blocks_v1(
    vllm_config: VllmConfig,
    layer_groups: list[list[str]],
    block_ids: list[list[int]],
    device: Optional[torch.device | str] = None,
) -> CacheBlocks:
    """
    return: [group_id, layer_id] -> (block_indices, k_cache_blocks, v_cahce_blocks)
    """
    if device is None:
       device = torch.accelerator.current_accelerator()
    cache_tensors = get_cache_tensors_v1(vllm_config, layer_groups)
    assert len(block_ids) == len(cache_tensors)
    cache_blocks = [ [] for _ in cache_tensors ]
    for cache, blocks, cache_tensor in \
      zip(cache_blocks, block_ids, cache_tensors):
        print("get block ids: ", block_ids)
        cache.extend(
            (
                blocks, 
                layer[0][blocks].to(device),
                layer[1][blocks].to(device),
            )
        for layer in cache_tensor
        )
    return cache_blocks

def set_cache_blocks_v1(
    vllm_config: VllmConfig,
    layer_groups: list[list[str]],
    cache_blocks: CacheBlocks,
):
    cache_tensors = get_cache_tensors_v1(vllm_config, layer_groups)
    for layer_tensors, layer_cache_blocks in zip(cache_tensors, cache_blocks):
        for layer_tensor, (block_indices, k_cache, v_cache) in \
          zip(layer_tensors, layer_cache_blocks):
            # TODO(arusuki): optimize cache block copy here
            layer_tensor[0][block_indices] = k_cache.to(layer_tensor.device)
            layer_tensor[1][block_indices] = v_cache.to(layer_tensor.device)

def get_layer_groups(core: EngineCore) -> list[list[str]]:
    assert isinstance(core.scheduler, Scheduler)
    kv_cache_config = core.scheduler.kv_cache_config
    return [
        group_spec.layer_names \
        for group_spec in kv_cache_config.kv_cache_groups
    ]

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
    layer_groups: list[list[str]],
):
    gpu_worker = worker.worker
    if not isinstance(gpu_worker, GPUWorker):
        raise ValueError(f"unsupported worker type: {type(gpu_worker)}")
    captured = runners[worker.rpc_rank]
    gpu_worker.model_runner.requests = captured.requests
    set_cache_blocks_v1(worker.vllm_config, layer_groups, captured.cache_blocks)

def rebuild_core_executor(core: EngineCore):
    def restore_uni_proc_executor(
        executor: UniProcExecutor,
        runners: list[CapturedGPUModelRunner],
    ):
        model_runner = executor.driver_worker.worker.model_runner
        assert isinstance(model_runner, GPUModelRunner)
        assert len(runners) == 1
        model_runner.requests = runners[0].requests
        layer_groups = get_layer_groups(core)
        set_cache_blocks_v1(core.vllm_config, layer_groups, runners[0].cache_blocks)

    def restore_multi_proc_executor(
        executor: MultiprocExecutor,
        runners: list[CapturedGPUModelRunner],
    ):
        layer_groups = get_layer_groups(core)
        return executor.collective_rpc(
            set_worker_kv_cache,
            args=(runners, layer_groups)
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
 
