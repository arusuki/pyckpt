import inspect
import sys
import threading
from _thread import LockType as LockType
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Thread, _active
from types import FrameType, NoneType
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import bytecode

from pyckpt import interpreter, objects
from pyckpt.analyzer import analyze_stack_top
from pyckpt.frame import (
    FunctionFrameCocoon,
    GeneratorFrameCocoon,
    LiveFrame,
    snapshot_from_frame,
)
from pyckpt.interpreter.frame import NullObject
from pyckpt.objects import Mapping

FrameCocoon = Union[FunctionFrameCocoon, GeneratorFrameCocoon]

_waiting_threads: Set[Thread] = set()

_original_lock_acquire = LockType.acquire


def _current_thread() -> Optional[Thread]:
    tid = threading.get_ident()
    return _active[tid] if tid in _active else None


def _lock_acquire(self, blocking: bool = True, timeout: float = -1):
    t = _current_thread()
    if t is None:
        return _original_lock_acquire(self, blocking, timeout)

    _waiting_threads.add(t)
    ret = _original_lock_acquire(self, blocking, timeout)
    _waiting_threads.discard(t)
    return ret


def _construct_waiting_threads(suspended: Dict[Thread, FrameType]):
    frames = sys._current_frames()
    return {t: frames[t.ident] for t in _waiting_threads if t not in suspended}


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
        self._handle = handle
        self._states = exception_states

        handle.__init__(target=self._evaluate)

    def _evaluate(self):
        leaf_frame: LiveFrame = self._leaf_frame.spawn()
        non_leaf_frames = [frame.spawn() for frame in self._non_leaf_frames]
        interpreter.restore_thread_state(self._states)

        ret, exc_states = leaf_frame.evaluate()
        for frame in non_leaf_frames:
            ret, exc_states = frame.evaluate(ret, exc_states)
        if exc_states is not None:
            # FIXME:
            raise NotImplementedError(
                f"frame evaluation ends with exception state: {exc_states[1]}"
            )

    @property
    def handle(self) -> Thread:
        return self._handle

    def evaluate(self, timeout=None):
        if self._resumed:
            raise RuntimeError("re-evaluating a thread")
        self._resumed = True
        self._handle.start()
        self._handle.join(timeout)


@dataclass
class ThreadCocoon:
    thread_id: int
    leaf_frame: FrameCocoon
    non_leaf_frames: List[FrameCocoon]
    exception_states: Any

    @staticmethod
    def extract_frames(
        frame: FrameType,
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

    def clone(
        self, object_table: Dict, persist_mapping: Optional[Dict[Type, Mapping]] = None
    ) -> "ThreadCocoon":
        def persist_thread(t: Thread):
            tid = id(t)
            if tid not in object_table:
                object_table[tid] = object.__new__(Thread)
            return tid

        pm = persist_mapping if persist_mapping else {}
        pm.update({Thread: persist_thread})
        if self.thread_id not in object_table:
            object_table[self.thread_id] = object.__new__(Thread)
        return objects.copy(self, objects=object_table, persist_mapping=pm)


def snapshot_from_thread(
    t: Thread,
    max_frames: Optional[int] = None,
    frame: Optional[FrameType] = None,
):
    """
    Either `(t is threading.current_thread()) == True` or called by op in THE_WORLD()`
    """
    if frame is None:
        frame = sys._current_frames()[t.ident]

    last_frame = snapshot_from_frame(frame, False, analyze_stack_top)
    if last_frame is None:  # return from LiveThread
        return None

    if frame.f_code is _lock_acquire.__code__:
        assert isinstance(last_frame, FunctionFrameCocoon)
        arg_names, *_, f_locals = inspect.getargvalues(frame)
        assert len(arg_names) == 3
        last_frame.stack.append(NullObject)
        last_frame.stack.append(_lock_acquire)
        last_frame.stack.extend(f_locals[n] for n in arg_names)
        last_frame.is_leaf = True
        last_frame.prev_instr_offset -= (
            1 + bytecode.ConcreteInstr("CALL", 0).use_cache_opcodes()
        )

    non_leaf_frames = ThreadCocoon.extract_frames(frame, max_frames)
    exception_states = interpreter.save_thread_state(t)
    return ThreadCocoon(id(t), last_frame, non_leaf_frames, exception_states)


class StarPlatinum:
    def __init__(
        self,
        operation: Callable[
            [Dict[Thread, FrameType], Dict[Thread, FrameType]], NoneType
        ],
        timeout=None,
    ):
        self._user = threading.current_thread()
        self._profile = threading.getprofile()
        self._threads: Dict[Thread, FrameType] = {}
        self._waiting_threads: Dict[Thread, FrameType] = {}
        self._all_stopped: Optional[Event] = Event()
        self._all_released: Optional[Event] = Event()
        self._running_cnt = 0
        self._timeout = timeout
        self._op = operation
        self._ret = None

        self.exc: Optional[Exception] = None

    @contextmanager
    def _reset_profile(self):
        try:
            yield
        except Exception as e:
            self.exc = e
        self._all_released.set()
        sys.setprofile(self._profile)

    def _reach(self, cnt: int):
        self._running_cnt -= cnt
        if self._running_cnt == 0:
            self._all_stopped.set()

    def profile_callback(self, frame: FrameType, _event, _arg):
        ident = threading.current_thread()
        with self._reset_profile():
            if ident is not self._user:
                self._threads[ident] = frame
                self._reach(1)
                self._all_released.wait()
                return None
            num_threads = sum(1 for _ in threading.enumerate())
            num_threads -= 1 + len(_waiting_threads)
            assert num_threads >= 0, "invalid thread count"
            self._reach(-num_threads)
            if not self._all_stopped.wait(self._timeout):
                raise RuntimeError("timeout on waiting thread to suspend")
            _wait = _construct_waiting_threads(self._threads)
            self._ret = self._op(self._threads, _wait)
            return None

    def THE_WORLD(self):
        def trap():
            pass

        interpreter.set_profile_all_threads(self.profile_callback)
        trap()
        if self.exc:
            raise self.exc
        return self._ret
