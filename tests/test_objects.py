import inspect
import io
import pickle
from queue import SimpleQueue
from threading import Thread
from typing import Generator

import pytest
import torch

from pyckpt.objects import (
    Pickler,
    copy,
    dump,
    load,
    load_untyped_storages,
    save_untyped_storages,
)


def test_dump_basic_object():
    buf = io.BytesIO()
    obj = {"a": 1, "b": 2}
    pickler = Pickler(buf)
    dump(pickler, obj)
    buf.seek(0)
    loaded = pickle.load(buf)
    assert loaded == obj


def test_dump_with_frame_type():
    buf = io.BytesIO()
    frame = inspect.currentframe()
    obj = {"frame": frame}
    pickler = Pickler(buf)
    dump(pickler, obj)
    buf.seek(0)
    loaded = pickle.load(buf)
    assert loaded["frame"] is None


def test_reduce_generator():
    def sample_generator():
        yield 1
        yield 2

    gen = sample_generator()
    assert next(gen) == 1
    buf = io.BytesIO()
    pickler = Pickler(buf)
    dump(pickler, gen)

    buf.seek(0)
    loaded = pickle.load(buf)
    assert isinstance(loaded, Generator)
    assert next(loaded) == 2
    assert next(gen) == 2
    with pytest.raises(StopIteration):
        next(loaded)
    with pytest.raises(StopIteration):
        next(gen)


def test_load_basic_object():
    buf = io.BytesIO()
    obj = {"a": 1, "b": 2}
    pickler = Pickler(buf)
    threads = dump(pickler, obj)
    buf.seek(0)
    loaded, _ = load(buf, threads)
    assert loaded == obj


def test_load_with_threads():
    buf = io.BytesIO()
    thread = Thread()
    obj = {"thread": thread}

    pickler = Pickler(buf)
    _threads = dump(pickler, obj)
    buf.seek(0)
    assert Thread in _threads
    assert id(thread) in _threads[Thread]

    loaded, objs = load(buf, _threads)
    assert "thread" in loaded
    assert id(thread) in objs
    thread_stub = objs[id(thread)]
    assert loaded["thread"] is thread_stub


def test_load_with_generator():
    def sample_generator():
        yield 1
        yield 2

    gen = sample_generator()
    assert next(gen) == 1
    buf = io.BytesIO()
    pickler = Pickler(buf)
    threads = dump(pickler, gen)

    buf.seek(0)
    loaded, _ = load(buf, threads)
    assert isinstance(loaded, Generator)
    assert next(loaded) == 2
    with pytest.raises(StopIteration):
        next(loaded)

def test_reduce_simple_queue():
    sq = SimpleQueue()
    sq.put(42)
    sq.put(43)

    buf = io.BytesIO()
    dump(buf, sq)

    buf.seek(0)
    loaded = pickle.load(buf)
    assert isinstance(loaded, SimpleQueue)
    assert loaded.get(block=False) == 42
    assert loaded.get(block=False) == 43

def test_save_tensor_storage_dump_load():
    x = torch.tensor(range(12)).reshape(3, 4)
    if torch.cuda.is_available():
        x = x.cuda()
    file = io.BytesIO()
    s = io.BytesIO()
    stores = dump(file, x)
    save_untyped_storages(s, stores["storage"])

    s.seek(0)
    file.seek(0)
    stores["storage"] = load_untyped_storages(s)
    new_x, _ = load(file, stores)

    assert torch.equal(x, new_x)
    assert x.device == new_x.device

def test_save_tensor_storage_copy():
    x = torch.tensor(range(12)).reshape(3, 4)
    x_slice = x[2]
    (new_x, new_x_slice), _ = copy((x, x_slice))

    assert new_x.untyped_storage() == \
      new_x_slice.untyped_storage()

    assert torch.equal(x, new_x)
    assert torch.equal(x_slice, new_x_slice)

def test_save_multiple_tensors():
    x = torch.tensor(range(12)).reshape(3, 4)
    y = torch.tensor(range(16)).reshape(4, 4)

    assert x.untyped_storage()._cdata != y.untyped_storage()._cdata

    file = io.BytesIO()
    s = io.BytesIO()
    stores = dump(file, (x, y))["storage"]
    assert len(stores) == 2

