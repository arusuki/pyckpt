import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import Any, Callable, Dict, List, Optional

import dill

from pyckpt import interpreter, platform
from pyckpt.analyzer import analyze_stack_top
from pyckpt.frame import FrameCocoon
from pyckpt.objects import (
    CheckpointRestoreContext,
    CRContextCocoon,
    ObjectCocoon,
    SnapshotContextManager,
    SpawnContextManager,
)


class LiveThread:
    @staticmethod
    def run_thread(run: Callable):
        run()

    def __init__(
        self,
        original_id: int,
        leaf_frame: FrameCocoon,
        non_leaf_frames: List[FrameCocoon],
        contexts: SpawnContextManager,
        states: Dict,
    ):
        for non_leaf in non_leaf_frames:
            assert not non_leaf.is_leaf, "pervious frames should be non-leaf frames"
        self._resumed = False
        self._leaf_frame = leaf_frame
        self._non_leaf_frames = non_leaf_frames
        self._result = Future()
        self._handle = threading.Thread(target=self._evaluate)
        self._contexts = contexts
        self._states = states

        # It's important that all ThreadCocoons should be spawned first
        # before start evaluating.
        contexts.register_object(original_id, self._handle)

    def _evaluate(self):
        leaf_frame = self._leaf_frame.spawn(self._contexts)
        non_leaf_frames = [
            frame.spawn(self._contexts) for frame in self._non_leaf_frames
        ]
        interpreter.restore_thread_state(self._states)

        ret, exc_states = leaf_frame.evaluate()
        for frame in reversed(non_leaf_frames):
            ret, exc_states = frame.evaluate(ret, exc_states)
        if exc_states is not None:
            # FIXME:
            raise NotImplementedError(
                f"frame evaluation ends with exception state: {exc_states[1]}"
            )
        self._result.set_result(ret)

    @property
    def handle(self) -> Thread:
        return self._handle

    def evaluate(self, timeout=None):
        if self._resumed:
            raise RuntimeError("re-evaluating a thread")
        self._resumed = True
        self._handle.start()
        return self._result.result(timeout=timeout)


def spawn_thread(original_id: int, mgr: SpawnContextManager):
    return mgr.retrieve_object(original_id)


def snapshot_thread(t: Thread, _mgr: Any):
    original_id = t.ident
    return ObjectCocoon(original_id, spawn_thread)


class ThreadContext(CheckpointRestoreContext):
    @staticmethod
    def spawn_method(_states):
        return ThreadContext()

    def snapshot(self, snapshot_ctxs: "SnapshotContextManager") -> "CRContextCocoon":
        return CRContextCocoon(ThreadContext.spawn_method, None)

    def register_snapshot_method(self, snapshot_ctxs: "SnapshotContextManager"):
        snapshot_ctxs.register_snapshot_method(Thread, snapshot_thread)

    def spawn(self, spawn_ctxs: "SpawnContextManager") -> "CheckpointRestoreContext":
        return

    def spawn_epilog(self, spawn_ctxs: "SpawnContextManager"):
        pass


@dataclass(frozen=True)
class ThreadCocoon:
    thread_id: int
    leaf_frame: FrameCocoon
    non_leaf_frames: List[FrameCocoon]
    thread_states: Dict

    @dataclass
    class ThreadStub:
        thread_id: Optional[int]

    @staticmethod
    def extract_frames(
        frame,
        max_frames: Optional[int],
        ctx_mgr: SnapshotContextManager,
    ) -> List[FrameCocoon]:
        non_leaf_frames = []
        current_frames = 1
        stop_code = {Thread.run.__code__}
        while frame.f_back is not None:
            if max_frames and current_frames >= max_frames:
                break
            frame = frame.f_back
            if frame.f_code in stop_code:
                break
            non_leaf_frames.append(
                FrameCocoon.snapshot_from_frame(
                    frame, False, analyze_stack_top, ctx_mgr
                )
            )
            current_frames += 1
        return non_leaf_frames

    @staticmethod
    def snapshot_from_thread(
        t: Thread,
        snapshot_ctxs: SnapshotContextManager,
        *,
        max_frames: Optional[int] = None,
    ):
        if Thread not in snapshot_ctxs.registry():
            raise ValueError("`Thread` class not in snapshot registry")
        is_other = threading.current_thread().ident != t.ident
        if is_other:
            platform.suspend_thread(t)

        frame = sys._current_frames()[t.ident]
        last_frame = FrameCocoon.snapshot_from_frame(
            frame, is_other, analyze_stack_top, snapshot_ctxs
        )
        if last_frame is None:  # return from LiveThread
            return None
        non_leaf_frames = ThreadCocoon.extract_frames(frame, max_frames, snapshot_ctxs)
        thread_state = interpreter.save_thread_state(t)
        return ThreadCocoon(t.ident, last_frame, non_leaf_frames, thread_state)

    def spawn(self, contexts: SpawnContextManager) -> LiveThread:
        live_thread = LiveThread(
            self.thread_id,
            self.leaf_frame,
            self.non_leaf_frames,
            contexts,
            self.thread_states,
        )
        return live_thread

    def clone(self) -> "ThreadCocoon":
        return dill.copy(self)
