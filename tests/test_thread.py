import threading
from typing import List, Optional

from pyckpt.objects import CRContextCocoon, SnapshotContextManager, SpawnContextManager
from pyckpt.thread import LiveThread, ThreadCocoon, ThreadContext


def test_thread_capture(capsys):
    c: Optional[ThreadCocoon] = None
    ctx_states: Optional[List[CRContextCocoon]] = None

    s = "hello_world"

    def test():
        nonlocal ctx_states
        nonlocal c

        snapshot_ctxs = SnapshotContextManager()
        thread_ctx = ThreadContext()
        snapshot_ctxs.register_context(thread_ctx)
        thread_cocoon = ThreadCocoon.snapshot_from_thread(
            threading.current_thread(), snapshot_ctxs
        )

        if thread_cocoon is not None:
            ctx_states = snapshot_ctxs.snapshot_contexts()
            c = thread_cocoon.clone()

        print(s)

    t = threading.Thread(target=test)
    t.start()
    t.join()

    result = capsys.readouterr()
    assert result.out.count(s) == 1
    assert isinstance(c, ThreadCocoon)
    assert isinstance(ctx_states, List)

    spawn_ctxs = SpawnContextManager.build_from_context_snapshot(ctx_states)
    live_thread: LiveThread = c.spawn(spawn_ctxs)
    live_thread.evaluate(timeout=1.0)
    result = capsys.readouterr()
    assert result.out.count(s) == 1


def test_thread_capture_with_exception(capsys):
    c: Optional[ThreadCocoon] = None
    ctx_states: Optional[List[CRContextCocoon]] = None

    s = "hello_world"

    def test():
        nonlocal c
        nonlocal ctx_states
        contexts = SnapshotContextManager()
        thread_ctx = ThreadContext()
        contexts.register_context(thread_ctx)
        try:
            raise RuntimeError("test")
        except RuntimeError:
            thread_cocoon = ThreadCocoon.snapshot_from_thread(
                threading.current_thread(), contexts
            )
        if thread_cocoon is not None:
            c = thread_cocoon.clone()
            ctx_states = contexts.snapshot_contexts()
        print(s)

    t = threading.Thread(target=test)
    t.start()
    t.join()

    result = capsys.readouterr()
    assert result.out.count(s) == 1

    assert c is not None
    spawn_ctx = SpawnContextManager.build_from_context_snapshot(ctx_states)
    live_thread: LiveThread = c.spawn(spawn_ctx)
    live_thread.evaluate(timeout=1.0)

    result = capsys.readouterr()
    assert result.out.count(s) == 1
