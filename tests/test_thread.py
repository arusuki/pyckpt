import threading
from types import FrameType
from typing import Optional, Set

from pyckpt.thread import (
    LiveThread,
    ThreadCocoon,
    ThreadSpawnContext,
    ThreadStubContext,
)


def test_thread_capture(capsys):
    c: Optional[ThreadCocoon] = None
    threads: Optional[Set[int]] = None

    s = "hello_world"

    def ret_none(*_):
        return None

    def test():
        nonlocal c
        nonlocal threads

        stub_registry = {}
        ThreadCocoon.register_stub(stub_registry)
        stub_registry[FrameType] = (ret_none, ret_none)
        stub_ctx = ThreadStubContext.new()

        contexts = {ThreadStubContext: stub_ctx}
        thread_cocoon = ThreadCocoon.from_thread(
            threading.current_thread(),
            stub_registry,
            contexts,
        )

        if thread_cocoon is not None:
            assert len(stub_ctx.contained_threads) != 0
            c = thread_cocoon.clone()
            threads = stub_ctx.contained_threads

        print(s)

    t = threading.Thread(target=test)
    t.start()
    t.join()

    result = capsys.readouterr()
    assert result.out.count(s) == 1

    assert c is not None
    assert threads is not None
    spawn_ctx = ThreadSpawnContext.new()
    live_thread: LiveThread = c.spawn({ThreadSpawnContext: spawn_ctx})
    live_thread.evaluate(timeout=1.0)

    result = capsys.readouterr()
    assert result.out.count(s) == 1


def test_thread_capture_with_exception(capsys):
    c: Optional[ThreadCocoon] = None
    threads: Optional[Set[int]] = None

    s = "hello_world"

    def ret_none(*_):
        return None

    def test():
        nonlocal c
        nonlocal threads

        stub_registry = {}
        ThreadCocoon.register_stub(stub_registry)
        stub_registry[FrameType] = (ret_none, ret_none)
        stub_ctx = ThreadStubContext.new()
        contexts = {ThreadStubContext: stub_ctx}
        try:
            raise RuntimeError("test")
        except RuntimeError:
            thread_cocoon = ThreadCocoon.from_thread(
                threading.current_thread(),
                stub_registry,
                contexts,
            )
        if thread_cocoon is not None:
            assert len(stub_ctx.contained_threads) != 0
            c = thread_cocoon.clone()
            threads = stub_ctx.contained_threads
        print(s)

    t = threading.Thread(target=test)
    t.start()
    t.join()

    result = capsys.readouterr()
    assert result.out.count(s) == 1

    assert c is not None
    assert threads is not None
    spawn_ctx = ThreadSpawnContext.new()
    live_thread: LiveThread = c.spawn({ThreadSpawnContext: spawn_ctx})
    live_thread.evaluate(timeout=1.0)

    result = capsys.readouterr()
    assert result.out.count(s) == 1
