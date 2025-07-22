import inspect
import os
from typing import Callable, ParamSpec
import random
import dill
import numpy as np
import torch

from torch import multiprocessing
from torch.multiprocessing import Process, Queue

from itertools import chain

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore


def print_attrs(obj):
    """get objects that fail to pickle"""
    for attr in dir(obj):
        if not attr.startswith("__") and not attr.endswith("__"):
            if attr == "_abc_impl":
                continue
            picklee = getattr(obj, attr)
            if inspect.ismethod(picklee):
                continue
            print("attr: ", attr)

_P = ParamSpec("_P")

class _TERMINATE: ...

TERMINATE = _TERMINATE()

def _process_func(func: Callable[_P, None], args: tuple, kwargs: dict):
    try:
        queue_type = type(Queue())
        func(*args, **kwargs)
    except Exception:
        for maybe_queue in chain(args, kwargs.values()):
            if isinstance(maybe_queue, queue_type):
                maybe_queue.put(TERMINATE)
        import traceback
        traceback.print_exc()
        os._exit(1)

def make_queue() -> Queue:
    ctx = multiprocessing.get_context("spawn") 
    return ctx.Queue()

def run_spawned(func: Callable[_P, None], *args: _P.args, **kwargs: _P.kwargs) -> Process:
    ctx = multiprocessing.get_context("spawn") 
    p = ctx.Process(target=_process_func, args=(func, args, kwargs))
    p.start()
    return p

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

def print_cache_tensor(core: EngineCore):
    manager = core.scheduler.kv_cache_manager
    assert isinstance(manager, KVCacheManager)
    assert isinstance(core.scheduler, Scheduler)
    assert len(manager.coordinator.single_type_managers) == 1

    full_attn_manager = manager.coordinator.single_type_managers[0]
    print("num_cached", full_attn_manager.num_cached_block)

    for req in core.scheduler.requests:
        print(
            "blocks: ", manager.coordinator.get_blocks(req)
        )
