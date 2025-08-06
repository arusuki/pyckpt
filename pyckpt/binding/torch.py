import os
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

import torch
from torch.distributed import ProcessGroup
from torch import distributed as dist

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


@dataclass
class DistributedEnv:
    envs: dict[str, str]
    init_args: tuple
    init_kw_args: dict


distributed_env: Optional[DistributedEnv] = None

_orig_torch_distributed_init_process_group = torch.distributed.init_process_group
_orig_torch_distributed_destroy_process_group = torch.distributed.destroy_process_group

DISTRIBUTED_ENVS = [
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
]


def patch_init_process_group(*args, **kw_args):
    global distributed_env

    if distributed_env or dist.is_initialized():
        raise RuntimeError("reinitialize torch.distributed process group")
    envs = {
        key: os.environ[key]
        for key in DISTRIBUTED_ENVS
        if os.environ.get(key) is not None
    }
    distributed_env = DistributedEnv(
        envs=envs,
        init_args=args,
        init_kw_args=kw_args,
    )
    _orig_torch_distributed_init_process_group(
        *args,
        **kw_args,
    )

def patch_destory_process_group(group: Optional[ProcessGroup] = None):
    global distributed_env
    if group:
        raise NotImplementedError("ProcessGroup support")

    _orig_torch_distributed_destroy_process_group(group)
    distributed_env = None


def save_distributed_group_states() -> Optional[DistributedEnv]:
    return distributed_env


def restore_distributed_group_states(states: DistributedEnv):
    global distributed_env

    if distributed_env or dist.is_initialized():
        raise RuntimeError("reinitialize torch.distributed process group")

    for env_key, value in states.envs.items():
        os.environ[env_key] = value

    distributed_env = states

    _orig_torch_distributed_init_process_group(*states.init_args, **states.init_kw_args)



def init():
    assert hasattr(dist, "init_process_group")
    assert hasattr(dist, "destroy_process_group")
    assert hasattr(torch.cuda, "get_device_properties")
    torch.cuda.get_device_properties = cuda_get_device_properties
    dist.init_process_group = patch_init_process_group
    dist.destroy_process_group = patch_destory_process_group
