from io import StringIO
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from threading import Thread, _active
import traceback
from types import FrameType
from typing import Any, Callable, NamedTuple, Optional, ParamSpec, TypeVar, cast
from urllib.parse import urlparse

import forbiddenfruit as patch
import msgpackrpc as rpc

from dill import Pickler as DataPickler
from dill import Unpickler as DataUnpickler
from pyckpt.frame import CALL_CODES, CaptureEvent, Frame, snapshot_frame
from pyckpt.interpreter import set_profile_all_threads, frame_lasti_opcode
from pyckpt.objects import PersistedObjects, Pickler, Unpickler

from _thread import LockType as LockType

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)

_local = threading.local()
_local._atomic = False


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
    def echo(self, msg: str):
        return msg

    def checkpoint(
        self, path: bytes, name: bytes, event_filter: Optional[list[CaptureEvent]]
    ):
        try:
            filter = None
            if event_filter:
                filter = set(event_filter)
            checkpoint_file_base = os.path.join(path, name)

            with (
                open(checkpoint_file_base + ".ckpt".encode(), "wb") as checkpoint_path,
                open(checkpoint_file_base + ".data".encode(), "wb") as data_path,
            ):
                checkpoint = Pickler(checkpoint_path)
                data = DataPickler(data_path)
                task_checkpoint(checkpoint, data, filter)
        except Exception as e:
            traceback.print_exception(e)
            raise


def parse_address(address):
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
        self.name = name
        self.threads: set[Thread] = set()
        self.blocking_threads: set[Thread] = set()
        self._server = rpc.Server(TaskDaemon())
        hostname, port = parse_address(daemon_addr)
        self._daemon = Thread(
            target=self._run_daemon,
            args=(rpc.Address(hostname, port),),
        )
        self._daemon.start()

    def _run_daemon(self, address: rpc.Address):
        logger.info("task daemon start listening on %s:%s", address.host, address.port)
        self._server.listen(address)
        self._server.start()

    def _shutdown(self):
        MAX_RETRY = 10
        TIME_WAIT = 0.1

        for _ in range(MAX_RETRY):
            self._server.close()
            self._server.stop()
            time.sleep(TIME_WAIT)
            self._daemon.join(timeout=TIME_WAIT)
            if not self._daemon.is_alive():
                return
            TIME_WAIT *= 1.7

        if self._daemon.is_alive():
            raise RuntimeError("timeout close server")

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


def get_task():
    if _task is None:
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
                with _patch_locks():
                    init_task(func.__name__, address)
                    get_task().register_current_thread()
                    return __mark_main(func, *args, **kwargs)
            except Exception as e:
                # panic
                logger.fatal("exception thorwn out of main function: %s", e)
                exit(-1)
            finally:
                if _task:
                    _task.threads.remove(threading.current_thread())
                    shutdown_task(wait_shutdown)

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
    if threading.current_thread() in task.threads:
        raise RuntimeError("call `checkpoint_task()` in task threads")
    original_profiler = sys.getprofile()
    assert original_profiler is None
    num_threads = len(task.threads)
    barrier = Barrier(num_threads)
    frames: CapturedThreads = {}
    profiled: set[Thread] = set()

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
            task = get_task()
            thread = threading.current_thread()
            if thread not in task.threads or thread in profiled:
                sys.setprofile(original_profiler)
                return
            if thread in task.blocking_threads:
                raise RuntimeError("profile blocking threads")

            if is_checkpoint_safe(frame):
                frames[thread] = (frame, CaptureEvent(event))
                with atomic():
                    barrier.wait(1, wait_arrive=False)
                profiled.add(thread)
                sys.setprofile(original_profiler)

        except Exception as e:
            # panic
            logger.fatal("profiler raise exception: %s", e)
            exit(-1)

    set_profile_all_threads(_profiler)
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
        checkpoint.dump(saved_task)
        persisted_objects = checkpoint.consume_persisted()
        data.dump(persisted_objects)
        sys.excepthook = excepthook

    return task_stop_insepct(checkpoint_profiler, event_filter)


def _current_thread() -> Optional[Thread]:
    tid = threading.get_ident()
    return _active.get(tid, None)


_original_lock_acquire = LockType.acquire


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
def _patch_locks():
    patch.curse(LockType, "acquire", _lock_acquire)
    try:
        yield
    finally:
        patch.curse(LockType, "acquire", _original_lock_acquire)


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
            worker.start()

        for worker in workers:
            # FIXME: only join main thread
            worker.join()

    _resume()


_MARK_STOP_BACKTRACE = set(marker.__code__ for marker in [__mark_main, Thread.run])
