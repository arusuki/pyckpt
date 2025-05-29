import threading
from typing import Dict, Generic, Optional, TypeVar

from pyckpt.thread import LiveThread, ThreadCocoon, snapshot_from_thread

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

    def set_value(self, new_value: T):
        self._value = new_value

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
    c: Optional[ThreadCocoon] = None
    objs: Optional[Dict] = {}
    flag = False
    multi_capture = SingletonValue(flag)

    s = "hello_world"
    exc = "executed"

    def test():
        nonlocal c

        thread_cocoon = snapshot_from_thread(threading.current_thread())
        if thread_cocoon is not None:
            c = thread_cocoon.clone(objs)

        if multi_capture.get_value():
            _thread_cocoon = snapshot_from_thread(threading.current_thread())
            if _thread_cocoon is not None:
                c = _thread_cocoon.clone(objs)
            print(exc)

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
    multi_capture.set_value(True)
    live_thread.evaluate(timeout=1.0)
    result = capsys.readouterr()
    assert result.out.count(s) == 1
    assert result.out.count(exc) == 1
