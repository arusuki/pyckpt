from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
import threading
import time
import traceback
from _thread import LockType as LockType
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from io import BytesIO, StringIO
from threading import Lock, Thread, _active, current_thread
from types import FrameType
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    ParamSpec,
    TypeVar,
    cast,
)
from urllib.parse import urlparse

import forbiddenfruit as patch
from dill import Pickler as DataPickler
from dill import Unpickler as DataUnpickler

from pyckpt import rpc
from pyckpt.frame import CALL_CODES, CaptureEvent, Frame, snapshot_frame
from pyckpt.interpreter import frame_lasti_opcode, set_profile_all_threads
from pyckpt.objects import PersistedObjects, Pickler, Unpickler, register_builtin

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)

_local = threading.local()
_local._atomic = False


class TaskProxy:
    def __reduce__(self):
        return TaskProxy, ()

    def __getattr__(self, name: str):
        return getattr(get_task(), name)


@contextmanager
def atomic():
    _atomic = is_atomic()
    _local._atomic = True
    try:
        yield
    finally:
        _local._atomic = _atomic


def is_atomic() -> bool:
    return getattr(_local, "_atomic", False)


class TaskDaemon:
    @staticmethod
    def echo(msg: str):
        return msg

    @staticmethod
    def checkpoint(path: str, name: str, event_filter: Optional[list[CaptureEvent]]):
        try:
            filter = None
            if event_filter:
                filter = set(event_filter)
            checkpoint_file_base = os.path.join(path, name)

            with (
                open(checkpoint_file_base + ".ckpt", "wb") as checkpoint_path,
                open(checkpoint_file_base + ".data", "wb") as data_path,
            ):
                checkpoint = Pickler(checkpoint_path)
                data = DataPickler(data_path)
                task_checkpoint(checkpoint, data, filter)
        except Exception as e:
            traceback.print_exception(e)
            raise


def parse_address(address) -> tuple[str, int]:
    if "://" not in address:
        address = "//" + address
    parsed = urlparse(address)
    host = parsed.hostname
    port = parsed.port
    return host, port


class Task:
    def __init__(
        self,
        name: str,
        daemon_addr: str,
    ):
        def _make_executor() -> ThreadPoolExecutor:
            executor = ThreadPoolExecutor(max_workers=1)
            # eagerly initialize the thread and wait for its start
            # so it's not tracked by task
            executor.submit(lambda: None).result()
            return executor

        self.name = name
        self.threads: set[Thread] = set()
        self.blocking_threads: set[Thread] = set()
        self._server = rpc.Server(TaskDaemon(), _make_executor())
        hostname, port = parse_address(daemon_addr)
        self._server.start(hostname, port)

    def __reduce__(self):
        return TaskProxy, ()

    def _shutdown(self):
        self._server.stop()

    def register_current_thread(self):
        thread = threading.current_thread()
        self.threads.add(thread)

    def update_threads(self):
        alive_threads = {thread for thread in self.threads if thread.is_alive()}
        finished = len(self.threads) - len(alive_threads)
        self.threads = alive_threads
        return finished


def generate_checkpoint_name():
    return f"{get_task().name}-{int(time.time())}"


_task: Optional[Task] = None


def get_task(maybe_none: bool = False):
    if _task is None and not maybe_none:
        raise RuntimeError("task not set, use @main to launch a task")
    return _task


def init_task(name: str, daemon_addr: str):
    global _task
    if _task is not None:
        raise RuntimeError("@main called multiple times")
    _task = Task(name, daemon_addr)


def shutdown_task(wait=True):
    global _task
    if threading.current_thread() in _task.threads:
        raise RuntimeError("worker tries to shutdown task")
    if wait:
        for thread in _task.threads:
            thread.join()
    _task._shutdown()
    _task = None


def __mark_main(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
    func(*args, **kwargs)


def main(address="localhost:9387", wait_shutdown=True):
    def decorate_main(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def _main(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                init_task(func.__name__, address)
                with _patch_builtin():
                    task = get_task()
                    task.register_current_thread()
                    logger.debug("task threads before enter main: %s", task.threads)
                    return __mark_main(func, *args, **kwargs)
            except Exception as e:
                # panic
                traceback.print_exception(e)
                logger.fatal("exception thorwn out of main function: %s", e)
                exit(-1)
            finally:
                if _task:
                    try:
                        _task.threads.discard(threading.current_thread())
                        shutdown_task(wait_shutdown)
                        logger.info("task shutdown")
                    except Exception as e:
                        tb = StringIO()
                        traceback.print_tb(e.__traceback__, file=tb)
                        logger.error(
                            f"exception in shutdown: {e}, traceback: {tb.getvalue()}"
                        )
                        exit(-1)

        return cast(Callable[P, T], _main)

    return decorate_main


class Barrier:
    def __init__(self, parties: int):
        self.parties = parties
        self.count = 0
        self.arrive = threading.Condition()
        self.leave = threading.Condition()

    def wait(self, count: int, wait_arrive: bool, timeout: Optional[float] = None):
        logger.debug(
            "%s: wait count %s, is worker: %s",
            threading.current_thread(),
            count,
            not wait_arrive,
        )
        with self.arrive:
            self.count += count
            if self.count > self.parties:
                # panic
                logger.fatal("barrier overflow")
                exit(-1)
            if wait_arrive and self.count < self.parties:
                self.arrive.wait(timeout)
            elif self.count == self.parties:
                self.arrive.notify_all()

        if not wait_arrive:
            with self.leave:
                self.leave.wait()

    def notify_leave(self):
        with self.leave:
            self.leave.notify_all()


CapturedThreads = dict[Thread, tuple[FrameType, int]]


def is_checkpoint_safe(current: FrameType) -> bool:
    current = current.f_back
    while current is not None:
        if current.f_code in _MARK_STOP_BACKTRACE:
            break
        opcode = frame_lasti_opcode(current)
        if opcode not in CALL_CODES:
            return False
        current = current.f_back
    return True


def task_stop_insepct(
    inspector: Callable[[CapturedThreads], T],
    event_filter: set[CaptureEvent] | None = None,
) -> T:
    if event_filter and len(event_filter) == 0:
        raise ValueError("empty event filter")
    task = get_task()
    current = threading.current_thread()
    if current in task.threads:
        raise RuntimeError("call `checkpoint_task()` in task threads")
    original_profiler = sys.getprofile()
    assert original_profiler is None
    num_threads = len(task.threads)
    barrier = Barrier(num_threads)
    logger.debug("wait threads: %s", task.threads)
    frames: CapturedThreads = {}
    profiled: set[Thread] = set()
    profiler_lock = Lock()

    def set_profiler(func: Callable):
        if current is threading.current_thread():
            if profiler_lock.locked():
                return
        try:
            _original_lock_acquire(profiler_lock)
            sys.setprofile(func)
        finally:
            profiler_lock.release()

    def _profiler(frame: FrameType, event: int, _arg: Any):
        try:
            if is_atomic():
                return
            if event in (
                CaptureEvent.C_RETURN,
                CaptureEvent.C_EXCEPTION,
                CaptureEvent.EXCEPTION,
            ):
                return
            if event_filter and event not in event_filter:
                return
            task = get_task(maybe_none=True)
            if task is None:
                set_profiler(original_profiler)
                return
            thread = threading.current_thread()
            if thread not in task.threads or thread in profiled:
                set_profiler(original_profiler)
                return
            if thread in task.blocking_threads:
                assert thread in profiled
                set_profiler(original_profiler)
                return

            if is_checkpoint_safe(frame):
                frames[thread] = (frame, CaptureEvent(event))
                with atomic():
                    barrier.wait(1, wait_arrive=False)
                profiled.add(thread)
                set_profiler(original_profiler)

        except Exception as e:
            # panic
            tb = StringIO()
            traceback.print_tb(e.__traceback__, file=tb)
            logger.fatal(
                "profiler<%s> raise exception: %s, %s", current_thread(), e, tb.getvalue()
            )
            exit(-1)

    _original_lock_acquire(profiler_lock)
    set_profile_all_threads(_profiler)
    profiler_lock.release()

    num_finished = task.update_threads()
    num_blocking = len(task.blocking_threads)
    barrier.wait(num_finished + num_blocking, wait_arrive=True)
    try:
        current_frames = sys._current_frames()
        # add thread frames that is blocking wait, e.g. Lock.acquire(blocking=True)
        frames.update(
            (
                (thread, (current_frames[thread.ident], CaptureEvent.C_CALL))
                for thread in task.blocking_threads
            )
        )
        profiled |= task.blocking_threads
        ret = inspector(frames)
    finally:
        barrier.notify_leave()
    return ret


@dataclass
class TaskThread:
    frames: list[Frame]


# id(thread) -> TaskThread
SavedTask = dict[int, TaskThread]


def capture_frame_backtrace(frame: FrameType, event: CaptureEvent) -> list[Frame]:
    logger.debug("leaf frame: %s, event: %s", frame, event)
    frames = [snapshot_frame(frame, event)]
    current = frame.f_back
    while current:
        if current.f_code in _MARK_STOP_BACKTRACE:
            break
        frames.append(snapshot_frame(current, CaptureEvent.INTERMEDIATE))
        if current.f_code is run_task_thread.__code__:
            break
        current = current.f_back
    return frames


def _debug_checkpoint(saved_task: SavedTask):
    for current in saved_task.values():
        for frame in current.frames:
            try:
                Pickler(BytesIO()).dump(frame)
            except Exception as e:
                logger.debug("error dumping frame: %s, exception: %s", frame, e)


def task_checkpoint(
    checkpoint: Pickler,
    data: DataPickler,
    event_filter: Optional[set[CaptureEvent]] = None,
):
    def checkpoint_profiler(frames: CapturedThreads):
        saved_task: SavedTask = {}
        for thread, (frame, event) in frames.items():
            saved_task[id(thread)] = TaskThread(
                frames=capture_frame_backtrace(frame, event)
            )
        excepthook = sys.excepthook
        sys.excepthook = sys.__excepthook__
        try:
            checkpoint.dump(saved_task)
        except Exception:
            _debug_checkpoint(saved_task)
            raise
        persisted_objects = checkpoint.consume_persisted()
        data.dump(persisted_objects)
        sys.excepthook = excepthook

    return task_stop_insepct(checkpoint_profiler, event_filter)


def _current_thread() -> Optional[Thread]:
    tid = threading.get_ident()
    return _active.get(tid, None)


_original_lock_acquire = register_builtin(LockType.acquire)


def _lock_acquire(self, blocking: bool = True, timeout: float = -1):
    thread = _current_thread()
    if not blocking or thread is None or is_atomic():
        return _original_lock_acquire(self, blocking, timeout)
    task = _task
    if task is None or thread not in task.threads:
        return _original_lock_acquire(self, blocking, timeout)

    task.blocking_threads.add(thread)
    ret = _original_lock_acquire(self, blocking, timeout)
    task.blocking_threads.discard(thread)
    return ret


@contextmanager
def _patch_builtin():
    _orignial_init = threading.Thread.__init__

    def patched_init(self, *args, **kwargs):
        task = get_task()
        if current_thread() not in task.threads:
            return _orignial_init(self, *args, **kwargs)
        _disable_patch = kwargs.pop("_disable_patch", False)
        _orignial_init(self, *args, **kwargs)
        if not _disable_patch:
            get_task().threads.add(self)

    patch.curse(LockType, "acquire", _lock_acquire)
    threading.Thread.__init__ = patched_init
    try:
        yield
    finally:
        patch.curse(LockType, "acquire", _original_lock_acquire)
        Thread.__init__ = _orignial_init


class LoadedCheckpoint(NamedTuple):
    task_threads: SavedTask
    persisted: PersistedObjects
    loaded_objects: dict[int, Any]


def load_checkpoint(
    checkpoint_path: str,
    name: str,
) -> LoadedCheckpoint:
    checkpoint_filename_base = os.path.join(checkpoint_path, name)

    with open(checkpoint_filename_base + ".data", "rb") as data_file:
        persisted = DataUnpickler(data_file).load()
    assert isinstance(persisted, PersistedObjects)

    with open(checkpoint_filename_base + ".ckpt", "rb") as checkpoint_file:
        unpickler = Unpickler(checkpoint_file, persisted)
        saved_task = unpickler.load()
        assert isinstance(saved_task, dict)
        loaded_objects = unpickler.get_loaded_objects()

    return LoadedCheckpoint(
        task_threads=saved_task,
        persisted=persisted,
        loaded_objects=loaded_objects,
    )


def run_task_thread(thread: TaskThread):
    frames = thread.frames
    if len(frames) == 0:
        raise ValueError("evaluate_frames: empty frames")
    frame = frames.pop(0)
    ret, exc_states = frame.evaluate()
    for idx, frame in enumerate(frames):
        ret, exc_states = frame.return_value(ret).evaluate(exc_states)
        frames[idx] = None
    if exc_states is not None:
        # FIXME(arusuki): currently, `_PyInterpreterFrame` is manually allocated.
        # CPython create frame objects on `_PyInterpreterFrame`, so when it's freed,
        # we are creating dangling references in frame objects in frame objects in traceback.
        # Fix this will need to look into how cpython handle such reference when
        # interpreter frame is poped.

        s = StringIO()
        exc = exc_states[1]
        # exc.__traceback__ = exc_states[2] # <-- This will cause segmentation fault
        traceback.print_exception(exc, file=s)
        # panic
        logger.fatal(
            "fatal: frame evaluation ends with exception state: %s, %s",
            exc_states[1],
            s.getvalue(),
        )
        exit(-1)


def resume_checkpoint(
    checkpoint: LoadedCheckpoint,
    address="localhost:9387",
    wait_shutdown=True,
):
    workers: list[Thread] = []
    for tid, task_thread in checkpoint.task_threads.items():
        if tid in checkpoint.loaded_objects:
            thread = checkpoint.loaded_objects[tid]
            assert isinstance(thread, Thread)
        else:
            thread = object.__new__(Thread)
        thread.__init__(target=run_task_thread, args=(task_thread,))
        workers.append(thread)

    @main(address=address, wait_shutdown=wait_shutdown)
    def _resume():
        for worker in workers:
            get_task().threads.add(worker)
            get_task().threads.discard(current_thread())

        for worker in workers:
            worker.start()

        for worker in workers:
            # FIXME: only join main thread
            worker.join()

    _resume()


_MARK_STOP_BACKTRACE = set(marker.__code__ for marker in [__mark_main, Thread.run])
