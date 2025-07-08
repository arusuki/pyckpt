from concurrent.futures import ThreadPoolExecutor
import inspect
import io
import pickle
from abc import ABC
from queue import SimpleQueue
from threading import Thread
from typing import Generator

import pytest

import pyckpt.objects as objects
from pyckpt.objects import Pickler, dump, load


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

