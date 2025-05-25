import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Union

from pyckpt import interpreter, objects, platform
from pyckpt.analyzer import analyze_stack_top
from pyckpt.frame import (
    FunctionFrameCocoon,
    GeneratorFrameCocoon,
    LiveFrame,
    snapshot_from_frame,
)

FrameCocoon = Union[FunctionFrameCocoon, GeneratorFrameCocoon]


class LiveThread:
    @staticmethod
    def run_thread(run: Callable):
        run()

    def __init__(
        self,
        handle: Thread,
        leaf_frame: FrameCocoon,
        non_leaf_frames: List[FrameCocoon],
        exception_states: Dict,
    ):
        for non_leaf in non_leaf_frames:
            assert not non_leaf.is_leaf, "pervious frames should be non-leaf frames"
        self._resumed = False
        self._leaf_frame = leaf_frame
        self._non_leaf_frames = non_leaf_frames
        self._result = Future()
        self._handle = handle
        self._states = exception_states

        handle.__init__(target=self._evaluate)

    def _evaluate(self):
        leaf_frame: LiveFrame = self._leaf_frame.spawn()
        non_leaf_frames = [frame.spawn() for frame in self._non_leaf_frames]
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
        self._handle.join(timeout)
        return self._result.result(timeout=timeout)


@dataclass
class ThreadCocoon:
    thread_id: int
    leaf_frame: FrameCocoon
    non_leaf_frames: List[FrameCocoon]
    exception_states: Any

    @staticmethod
    def extract_frames(
        frame,
        max_frames: Optional[int],
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
            non_leaf_frames.append(snapshot_from_frame(frame, False, analyze_stack_top))
            current_frames += 1
        return non_leaf_frames

    def spawn(self, handle: Thread) -> LiveThread:
        live_thread = LiveThread(
            handle,
            self.leaf_frame,
            self.non_leaf_frames,
            self.exception_states,
        )
        return live_thread

    def clone(self, object_table: Dict) -> "ThreadCocoon":
        def persist_thread(t: Thread):
            tid = id(t)
            if tid not in object_table:
                object_table[tid] = object.__new__(Thread)
            return tid

        if self.thread_id not in object_table:
            object_table[self.thread_id] = object.__new__(Thread)
        mapping = {Thread: persist_thread}
        return objects.copy(self, objects=object_table, persist_mapping=mapping)


def snapshot_from_thread(t: Thread, max_frames: Optional[int] = None):
    is_other = threading.current_thread().ident != t.ident
    if is_other:
        platform.suspend_thread(t)

    frame = sys._current_frames()[t.ident]
    last_frame = snapshot_from_frame(frame, is_other, analyze_stack_top)
    if last_frame is None:  # return from LiveThread
        return None
    non_leaf_frames = ThreadCocoon.extract_frames(frame, max_frames)
    exception_states = interpreter.save_thread_state(t)
    return ThreadCocoon(id(t), last_frame, non_leaf_frames, exception_states)
