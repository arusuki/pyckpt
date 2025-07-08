from io import BytesIO

import pytest
import torch.nn as nn
import torch

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

