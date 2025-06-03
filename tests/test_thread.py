import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from time import sleep
from types import FrameType
from typing import Callable, Dict, Generic, Optional, TypeVar

import forbiddenfruit as patch

from pyckpt.thread import (
    LiveThread,
    LockType,
    StarPlatinum,
    ThreadCocoon,
    _lock_acquire,
    _waiting_threads,
    snapshot_from_thread,
)

T = TypeVar("T")


class SingletonValue(Generic[T]):
    _instance: Optional["SingletonValue"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonValue, cls).__new__(cls)
        return cls._instance

    def __init__(self, value: T):
        if not SingletonValue._initialized:
            self._value = value
            SingletonValue._initialized = True

    def set_value(self, value: T):
        self._value = value

    def get_value(self) -> T:
        return self._value


def test_thread_capture(capsys):
    c: Optional[ThreadCocoon] = None
    objs: Optional[Dict] = {}

    s = "hello_world"

    def test():
        nonlocal c

        thread_cocoon = snapshot_from_thread(threading.current_thread())
        if thread_cocoon is not None:
            c = thread_cocoon.clone(objs)
        print(s)

    t = threading.Thread(target=test)
    t.start()
    t.join()

    result = capsys.readouterr()
    assert result.out.count(s) == 1
    assert isinstance(c, ThreadCocoon)
    assert isinstance(objs, Dict)

    assert id(t) in objs
    live_thread: LiveThread = c.spawn(objs[id(t)])
    live_thread.evaluate(timeout=1.0)
    result = capsys.readouterr()
    assert result.out.count(s) == 1


def test_thread_capture_with_exception(capsys):
    c: Optional[ThreadCocoon] = None
    objs: Optional[Dict] = {}

    s = "hello_world"

    def test():
        nonlocal c
        nonlocal objs
        try:
            raise RuntimeError("test")
        except RuntimeError:
            thread_cocoon = snapshot_from_thread(threading.current_thread())
        if thread_cocoon is not None:
            c = thread_cocoon.clone(objs)
        print(s)

    t = threading.Thread(target=test)
    t.start()
    t.join()

    result = capsys.readouterr()
    assert result.out.count(s) == 1

    assert c is not None
    live_thread: LiveThread = c.spawn(objs[c.thread_id])
    live_thread.evaluate(timeout=1.0)

    result = capsys.readouterr()
    assert result.out.count(s) == 1


def test_thread_multiple_captures(capsys):
    @dataclass
    class ReturnValue:
        cocoon: Optional[ThreadCocoon]
        multi_capture: bool

    objs: Optional[Dict] = {}
    ret = SingletonValue(ReturnValue(None, False))

    s = "hello_world"
    exc = "executed"

    def test():
        nonlocal ret

        thread_cocoon = snapshot_from_thread(threading.current_thread())
        if thread_cocoon is not None:
            cocoon = thread_cocoon.clone(objs)
            ret.get_value().cocoon = cocoon

        if ret.get_value().multi_capture:
            _thread_cocoon = snapshot_from_thread(threading.current_thread())
            if _thread_cocoon is not None:
                ret.get_value().cocoon = _thread_cocoon.clone(objs)
            print(exc)

        print(s)

    t = threading.Thread(target=test)
    t.start()
    t.join()

    result = capsys.readouterr()
    assert result.out.count(s) == 1
    assert isinstance(ret.get_value().cocoon, ThreadCocoon)
    assert isinstance(objs, Dict)

    assert id(t) in objs
    cocoon = ret.get_value().cocoon
    live_thread: LiveThread = cocoon.spawn(objs[id(t)])

    ret.get_value().multi_capture = True
    live_thread.evaluate(timeout=1.0)
    result = capsys.readouterr()
    assert result.out.count(s) == 1
    assert result.out.count(exc) == 1

    cocoon = ret.get_value().cocoon
    live_thread = cocoon.spawn(objs[id(live_thread.handle)])

    ret.get_value().multi_capture = False
    live_thread.evaluate(timeout=None)
    result = capsys.readouterr()
    assert result.out.count(s) == 1


@contextmanager
def _guard(start: threading.Event, f: Callable):
    try:
        yield
    finally:
        start.set()
        f()


@contextmanager
def lock_guard(lock: LockType):
    try:
        yield
    finally:
        if lock.locked():
            lock.release()


@contextmanager
def _replace(obj, attr, val):
    orig = getattr(obj, attr)
    patch.curse(obj, attr, val)
    try:
        yield
    finally:
        patch.curse(obj, attr, orig)


def test_patch_lock_acquire():
    _lock = threading.Lock()

    def foo():
        with _replace(LockType, "acquire", _lock_acquire):
            _lock.acquire(timeout=1.0)

    assert _lock.acquire(timeout=1.0)

    with lock_guard(_lock):
        t = threading.Thread(target=foo)
        t.start()
        MAX_RETRY = 10
        captured = False
        for _ in range(MAX_RETRY):
            if len(_waiting_threads) != 0:
                captured = True
                break
            sleep(0.1)
        assert captured
        assert t in _waiting_threads
        f = sys._current_frames()[t.ident]
        assert "_lock_acquire" in f.f_code.co_name
        assert f.f_back is not None
        f = f.f_back
        assert "foo" in f.f_code.co_name
    t.join()
    assert hasattr(LockType, "acquire")


def test_stop_the_world():
    start = threading.Event()
    finish = False

    def wait_thread():
        start.wait()

    def noop():
        pass

    def running_thread():
        while not finish:
            noop()

    t = threading.Thread(target=wait_thread)
    tr = threading.Thread(target=running_thread)

    def op(
        suspended_threads: Dict[threading.Thread, FrameType],
        waiting_threads: Dict[threading.Thread, FrameType],
    ):
        assert len(suspended_threads) == 1
        assert len(waiting_threads) == 1
        assert t in waiting_threads
        assert tr in suspended_threads
        frame = next(iter(suspended_threads.values()))
        assert frame.f_code in (running_thread.__code__, noop.__code__)
        frame = next(iter(waiting_threads.values()))
        assert frame.f_code is _lock_acquire.__code__

    with _replace(LockType, "acquire", _lock_acquire):
        t.start()
        tr.start()

        def finish_run():
            nonlocal finish
            finish = True

        sleep(0.1)
        with _guard(start, finish_run):
            StarPlatinum(op, timeout=1.0).THE_WORLD()

        start.set()
        finish = True

        t.join()
        tr.join()


def test_star_platinum_capture_waiting(capsys):
    start = threading.Event()
    objs = {}

    def no_op():
        pass

    def thread_func():
        start.wait(1.0)
        print("executed")

    def op(_, waiting_threads: Dict[threading.Thread, FrameType]):
        assert len(waiting_threads) == 1
        t, frame = next(iter(waiting_threads.items()))
        c = snapshot_from_thread(t, frame=frame)
        return c.clone(objs)

    t = threading.Thread(target=thread_func)

    with _replace(LockType, "acquire", _lock_acquire):
        t.start()
        with _guard(start, no_op):
            c = StarPlatinum(op, timeout=1.0).THE_WORLD()
        assert isinstance(c, ThreadCocoon)
        t.join()

    result = capsys.readouterr()
    assert result.out.count("executed") == 1

    f = c.non_leaf_frames[1]
    event: Optional[threading.Event] = None
    for i in f.nlocals:
        if isinstance(i, threading.Event):
            event = i
    assert event is not None
    event.set()
    live_thread: LiveThread = c.spawn(objs[c.thread_id])
    live_thread.evaluate(timeout=1.0)

    result = capsys.readouterr()
    assert result.out.count("executed") == 1
