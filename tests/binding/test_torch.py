import os
import socket
from io import BytesIO
from multiprocessing import Process
from typing import Callable, ParamSpec

import numpy
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist

from pyckpt import objects
from pyckpt.binding import torch as patch_torch
from pyckpt.binding.torch import (
    cuda_get_device_properties,
    restore_distributed_group_states,
    save_distributed_group_states,
)
from tests.utils import run_spawned
from torch.nn.parallel import DistributedDataParallel as DDP

from tests.utils import dump, load, copy

def join_safe(process: Process):
    process.join()
    assert process.exitcode == 0


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def init_distributed():
    port = find_free_port()
    init_method = f"tcp://localhost:{port}"

    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=torch.cuda.device_count(),
        rank=0,
    )


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def test_torch_simple_module():
    m = SimpleNN(10, 10, 10)
    buffer = BytesIO()
    p = dump(buffer, m)
    buffer.seek(0)
    nm, _ = load(buffer, p)
    assert isinstance(nm, SimpleNN)

    if torch.cuda.is_available():
        m = m.cuda()
    buffer.seek(0)
    p = dump(buffer, m)
    buffer.seek(0)
    nm, _ = load(buffer, p)
    assert isinstance(nm, SimpleNN)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_torch_cuda_device_property():
    property = cuda_get_device_properties()
    buffer = BytesIO()
    dump(buffer, property)


def test_torch_tensor_to_numpy():
    t = torch.tensor(range(12)).reshape(3, 4)
    n = t.numpy()
    assert torch.is_tensor(n.base)

    t[2][2] = 114
    assert n[2][2] == 114

    n[2][2] = 115
    assert t[2][2] == 115

    slice = n[2]
    slice[2] = 116
    assert t[2][2] == 116

    objects.get_leaf_base(slice)[2][2] = 117
    assert t[2][2] == 117

    assert (
        objects.get_leaf_base(slice).untyped_storage().data_ptr()
        == t.untyped_storage().data_ptr()
    )


def test_torch_dump_tensor_numpy():
    t = torch.tensor(range(12)).reshape(3, 4)
    n = t.numpy()
    assert torch.is_tensor(n.base)
    assert n.base.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()

    (new_t, new_n), _ = copy((t, n))
    assert isinstance(new_t, torch.Tensor)
    assert isinstance(new_n, numpy.ndarray)
    assert torch.is_tensor(new_n.base)
    assert new_n.base.untyped_storage().data_ptr() == new_t.untyped_storage().data_ptr()


def _init_distributed(
    master_addr: str,
    master_port: str,
    world_size: int,
    rank: int,
):
    patch_torch.init()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

_P = ParamSpec("_P")

def _run_distributed_task(task: Callable[_P, None], world_size: int):
    master_addr = "localhost"
    master_port = str(find_free_port())
    workers = []
    for rank in range(world_size):
        workers.append(
            run_spawned(
                task,
                master_addr,
                master_port,
                world_size,
                rank,
            )
        )
    for w in workers:
        join_safe(w)


def _simple_distributed_task(
    master_addr: str,
    master_port: str,
    world_size: int,
    rank: int,
):
    _init_distributed(master_addr, master_port, world_size, rank)
    dist.init_process_group("gloo")
    x = torch.tensor(rank)
    dist.all_reduce(x)
    assert x == sum(range(world_size))
    states = save_distributed_group_states()
    dist.destroy_process_group()

    assert states is not None
    restore_distributed_group_states(states)
    x = torch.tensor(rank)
    dist.all_reduce(x)
    assert x == sum(range(world_size))
    dist.destroy_process_group()


def test_torch_distributed_simple():
    _run_distributed_task(_simple_distributed_task, 8)

def _distributed_ddp_simple(
    master_addr: str,
    master_port: str,
    world_size: int,
    rank: int,
):
    _init_distributed(master_addr, master_port, world_size, rank)
    dist.init_process_group("gloo")
    model = DDP(SimpleNN(10, 10, 10))
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = model(torch.randn(20, 10))
    labels = torch.randn(20, 10)
    loss_fn(outputs, labels).backward()
    optimizer.step()


def test_torch_distributed_ddp():
    _run_distributed_task(_distributed_ddp_simple, 4)

