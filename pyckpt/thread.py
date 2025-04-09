import sys
import threading
from concurrent.futures import Future
from threading import Thread
from typing import Callable, Dict, List, Optional, Set, Type

import dill
from attr import dataclass

from pyckpt import interpreter, platform
from pyckpt.analyzer import analyze_stack_top
from pyckpt.frame import FrameCocoon


class LiveThread:
    @staticmethod
    def run_thread(run: Callable):
        run()

    def __init__(
        self,
        original_id: int,
        leaf_frame: FrameCocoon,
        non_leaf_frames: List[FrameCocoon],
        contexts: Dict,
        states: Dict,
    ):
        for non_leaf in non_leaf_frames:
            assert not non_leaf.is_leaf, "pervious frames should be non-leaf frames"
        self._resumed = False
        self._leaf_frame = leaf_frame
        self._non_leaf_frames = non_leaf_frames
        self._result = Future()
        self._handle = threading.Thread(target=self._evaluate)
        self._original_id = original_id
        self._contexts = contexts
        self._states = states

        spawn_context = contexts[ThreadSpawnContext]
        spawn_context.thread_table[self._original_id] = self._handle

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


@dataclass(frozen=True)
class ThreadStubContext:
    contained_threads: Set[int]

    @staticmethod
    def new():
        return ThreadStubContext(set())

    @staticmethod
    def get_contained_threads(contexts: Dict[int, "ThreadStubContext"]):
        if ThreadStubContext not in contexts:
            raise ValueError("contexts not contain `ThreadStubContext`")
        return contexts[ThreadStubContext].contained_threads


@dataclass(frozen=True)
class ThreadSpawnContext:
    thread_table: Dict[int, Thread]

    @staticmethod
    def new():
        return ThreadSpawnContext({})


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
    def register_stub(stub_registry: Dict):
        stub_registry[Thread] = (
            ThreadCocoon.stub_objects,
            ThreadCocoon.get_real_objects,
        )

    @staticmethod
    def stub_objects(t: Thread, contexts: Dict[Type, ThreadStubContext]):
        context = contexts[ThreadStubContext]
        thread_id = t.ident
        context.contained_threads.add(thread_id)
        return ThreadCocoon.ThreadStub(thread_id)

    @staticmethod
    def get_real_objects(stub: ThreadStub, contexts: Dict[Type, ThreadSpawnContext]):
        context = contexts[ThreadSpawnContext]
        return context.thread_table[stub.thread_id]

    @staticmethod
    def from_thread(
        t: Thread,
        stub_registry: Dict,
        contexts: Dict,
        *,
        max_frames: Optional[int] = None,
    ):
        if ThreadStubContext not in contexts:
            raise ValueError("context not contain `ThreadStubContext`")
        if Thread not in stub_registry:
            raise ValueError("`Thread` class not in stub registry")

        is_other = threading.current_thread().ident != t.ident
        if is_other:
            platform.suspend_thread(t)

        frame = sys._current_frames()[t.ident]
        non_leaf_frames = []
        last_frame = FrameCocoon.from_frame(
            frame, is_other, analyze_stack_top, stub_registry, contexts
        )
        if last_frame is None:  # return from LiveThread
            return None
        non_leaf_frames.append(last_frame)  # return from normal snapshot()
        current_frames = 1
        stop_code = {Thread.run.__code__}
        while frame.f_back is not None:
            if max_frames:
                current_frames += 1
                if current_frames >= max_frames:
                    break
            frame = frame.f_back
            if frame.f_code in stop_code:
                break
            non_leaf_frames.append(
                FrameCocoon.from_frame(
                    frame, False, analyze_stack_top, stub_registry, contexts
                )
            )
            if frame.f_code == ThreadCocoon.from_thread.__code__:
                break  # avoid redundant frames
        thread_state = interpreter.save_thread_state(t)
        return ThreadCocoon(t.ident, last_frame, non_leaf_frames, thread_state)

    def spawn(self, contexts: Dict[str, ThreadSpawnContext]) -> LiveThread:
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
