import inspect
import io
import pickle
from queue import SimpleQueue
from threading import Thread
from typing import Generator

import pytest

from pyckpt.objects import dump, load, create_pickler


def test_dump_basic_object():
    buf = io.BytesIO()
    obj = {"a": 1, "b": 2}
    pickler = create_pickler(buf)
    dump(pickler, obj)
    buf.seek(0)
    loaded = pickle.load(buf)
    assert loaded == obj


def test_dump_with_frame_type():
    buf = io.BytesIO()
    frame = inspect.currentframe()
    obj = {"frame": frame}
    pickler = create_pickler(buf)
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
    pickler = create_pickler(buf)
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
    pickler = create_pickler(buf)
    threads = dump(pickler, obj)
    buf.seek(0)
    loaded = load(buf, threads)
    assert loaded == obj


def test_load_with_threads():
    buf = io.BytesIO()
    thread = Thread(target=lambda: None)
    thread.start()
    thread.join()
    obj = {"thread": thread}
    thread_ids = set()

    def persist_thread(t: Thread):
        tid = id(t)
        thread_ids.add(tid)
        return tid

    persist_mapping = {Thread: persist_thread}
    pickler = create_pickler(buf, persist_mapping)
    _threads = dump(pickler, obj)
    buf.seek(0)
    assert id(thread) in thread_ids

    thread_stub = 42
    objs = {id(thread): thread_stub}
    loaded = load(buf, objs)
    assert "thread" in loaded
    assert loaded["thread"] is thread_stub


def test_load_with_generator():
    def sample_generator():
        yield 1
        yield 2

    gen = sample_generator()
    assert next(gen) == 1
    buf = io.BytesIO()
    pickler = create_pickler(buf)
    threads = dump(pickler, gen)

    buf.seek(0)
    loaded = load(buf, threads)
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

