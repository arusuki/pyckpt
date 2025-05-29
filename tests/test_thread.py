import threading
from typing import Dict, Optional, Any, Type
from dataclasses import dataclass
from pyckpt.thread import LiveThread, ThreadCocoon, snapshot_from_thread


class SoleObjects:
    _instance: Optional["SoleObjects"] = None
    _original_id: Optional[int] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SoleObjects, cls).__new__(cls)
            cls._original_id = id(cls._instance)
        return cls._instance

    def __init__(self, value: Any = None, value_type: Type = None):
        if not SoleObjects._initialized:
            self._value = value
            self._type = value_type
            SoleObjects._initialized = True

    def set_value(self, new_value: Any):
        self._value = new_value

    def get_value(self) -> Any:
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
    flag: bool = False
    multi_capture: SoleObjects = SoleObjects(flag, type(flag))

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
