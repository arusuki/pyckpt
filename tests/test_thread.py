import threading
from types import FrameType
from typing import Any, Optional

from pyckpt.objects import ObjectCocoon, SnapshotContextManager, SpawnContextManager
from pyckpt.thread import LiveThread, ThreadCocoon


def test_thread_capture(capsys):
    c: Optional[ThreadCocoon] = None

    s = "hello_world"

    def ret_none(*_):
        return None

    def snapshot_none(obj: Any, _):
        return ObjectCocoon(None, ret_none)

    def test():
        nonlocal c

        registry = {}
        ThreadCocoon.register_snapshot_hooks(registry)
        registry[FrameType] = snapshot_none
        snapshot_ctxs = SnapshotContextManager()

        thread_cocoon = ThreadCocoon.from_thread(
            threading.current_thread(),
            registry,
            snapshot_ctxs,
        )

        if thread_cocoon is not None:
            c = thread_cocoon.clone()

        print(s)

    t = threading.Thread(target=test)
    t.start()
    t.join()

    result = capsys.readouterr()
    assert result.out.count(s) == 1
    assert isinstance(c, ThreadCocoon)
    spawn_ctxs = SpawnContextManager()
    live_thread: LiveThread = c.spawn(spawn_ctxs)
    live_thread.evaluate(timeout=1.0)
    result = capsys.readouterr()
    assert result.out.count(s) == 1


def test_thread_capture_with_exception(capsys):
    c: Optional[ThreadCocoon] = None

    s = "hello_world"

    def ret_none(*_):
        return None

    def snapshot_none(obj: Any, _):
        return ObjectCocoon(None, ret_none)

    def test():
        nonlocal c

        stub_registry = {}
        ThreadCocoon.register_snapshot_hooks(stub_registry)
        stub_registry[FrameType] = snapshot_none
        contexts = SnapshotContextManager()
        try:
            raise RuntimeError("test")
        except RuntimeError:
            thread_cocoon = ThreadCocoon.from_thread(
                threading.current_thread(),
                stub_registry,
                contexts,
            )
        if thread_cocoon is not None:
            c = thread_cocoon.clone()
        print(s)

    t = threading.Thread(target=test)
    t.start()
    t.join()

    result = capsys.readouterr()
    assert result.out.count(s) == 1

    assert c is not None
    spawn_ctx = SpawnContextManager()
    live_thread: LiveThread = c.spawn(spawn_ctx)
    live_thread.evaluate(timeout=1.0)

    result = capsys.readouterr()
    assert result.out.count(s) == 1
