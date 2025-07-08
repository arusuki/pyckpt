from typing import Callable, Generic, TypeVar
import torch
from pyckpt.objects import reduce_as_none

def dispatch_table():
    return {
        torch.Module: reduce_as_none,
    }

T = TypeVar("T")

Constructor = Callable[[], T]

class ProxyObject(Generic[T]):

    def __init__(self, value: T, consturctor: Constructor):
        self._value = value
        self._ckpt_constructor = consturctor

    @staticmethod
    def construct_proxy(constructor: Constructor):
        return ProxyObject(constructor(), constructor)
    
    def __reduce__(self):
        return ProxyObject.construct_proxy, (self._ckpt_constructor,)

    def __getattr__(self, name: str):
        return getattr(self._value, name)

_orig_cuda_get_device_properties = torch.cuda.get_device_properties

def cuda_get_device_properties(device=None):

    def constructor():
        return _orig_cuda_get_device_properties(device)

    prop = _orig_cuda_get_device_properties(device)
    proxy = ProxyObject(prop, constructor)
    return proxy

def init():
    torch.cuda.get_device_properties = cuda_get_device_properties

