import threading
from typing import Dict, Optional

from pyckpt.thread import LiveThread, ThreadCocoon, snapshot_from_thread


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
