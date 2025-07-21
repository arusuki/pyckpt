from io import BytesIO

import numpy
import pytest
import torch
import torch.nn as nn

from pyckpt import objects
from pyckpt.binding.torch import cuda_get_device_properties


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
    p = objects.dump(buffer, m)
    buffer.seek(0)
    nm, _ = objects.load(buffer, p)
    assert isinstance(nm, SimpleNN)

    if torch.cuda.is_available():
        m = m.cuda()
    buffer.seek(0)
    p = objects.dump(buffer, m)
    buffer.seek(0)
    nm, _ = objects.load(buffer, p)
    assert isinstance(nm, SimpleNN)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_torch_cuda_device_property():
    property = cuda_get_device_properties()
    buffer = BytesIO()
    objects.dump(buffer, property)

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

    assert objects.get_leaf_base(slice).untyped_storage().data_ptr() == \
        t.untyped_storage().data_ptr()

def test_torch_dump_tensor_numpy():
    t = torch.tensor(range(12)).reshape(3, 4)
    n = t.numpy()
    assert torch.is_tensor(n.base)
    assert n.base.untyped_storage().data_ptr() == \
            t.untyped_storage().data_ptr()
    
    (new_t, new_n), _ = objects.copy((t, n))
    assert isinstance(new_t, torch.Tensor)
    assert isinstance(new_n, numpy.ndarray)
    assert torch.is_tensor(new_n.base)
    assert new_n.base.untyped_storage().data_ptr() == \
            new_t.untyped_storage().data_ptr()

