import inspect
import os
from typing import Callable, ParamSpec

from torch import multiprocessing
from torch.multiprocessing import Process, Queue

from itertools import chain


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
