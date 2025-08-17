import socket
from _thread import LockType as LockType
from threading import Event, Lock, Thread
from typing import Optional

import pyckpt.rpc as rpc
import py
import pytest

import pyckpt.task as task
from pyckpt.frame import FunctionFrame
from pyckpt.task import (
    Barrier,
    CapturedThreads,
    TaskThread,
    get_task,
    load_checkpoint,
    main,
    resume_checkpoint,
    task_stop_insepct,
)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def test_task_barrier():
    barrier = Barrier(2)

    def worker():
        barrier.wait(2, wait_arrive=False)

    t1 = Thread(target=worker)
    t1.start()
    barrier.wait(0, wait_arrive=True, timeout=1)
    barrier.notify_leave()
    t1.join()


def test_task_init():
    exception: Optional[Exception] = None

    @main(f"localhost:{find_free_port()}")
    def task_function():
        nonlocal exception
        try:
            task = get_task()
            assert len(task.threads) == 1
        except Exception as e:
            exception = e

    worker = Thread(target=task_function)
    worker.start()
    worker.join()
    assert exception is None


def test_task_stop_inspect():
    def inspector(threads: CapturedThreads):
        return len(threads) == 1

    lock = Lock()
    start_event = Event()
    assert lock.acquire(blocking=False)

    @main(f"localhost:{find_free_port()}")
    def task_function():
        start_event.set()
        while not lock.acquire(blocking=False):
            pass
        print("worker exit main")

    worker = Thread(target=task_function)
    worker.start()
    assert start_event.wait()
    try:
        ret = task_stop_insepct(inspector)
    finally:
        lock.release()
        worker.join()
    assert ret


def test_task_patch_lock_inspect():
    def inspector(threads: CapturedThreads):
        return len(threads) == 1

    event = Event()
    start_event = Event()

    free_port = find_free_port()

    @main(f"localhost:{free_port}")
    def task_function():
        start_event.set()
        event.wait()
        print("hello world")

    server = Thread(target=task_function)
    server.start()
    start_event.wait(timeout=1.0)
    try:
        ret = task_stop_insepct(inspector)
    finally:
        event.set()
        server.join()

    assert ret


def test_task_daemon_checkpoint(tmpdir: py.path.local):
    lock = Lock()
    start_event = Event()
    assert lock.acquire(blocking=False)

    free_port = find_free_port()

    def another_task():
        while not lock.acquire(blocking=False):
            pass
        lock.release()


    @main(f"localhost:{free_port}")
    def task_function():
        another = Thread(target=another_task)
        another.start()
        start_event.set()
        while not lock.acquire(blocking=False):
            pass
        lock.release()

    server = Thread(target=task_function)
    server.start()
    start_event.wait()
    try:
        client = rpc.Client()
        client.connect("localhost", free_port)
        filename = task.generate_checkpoint_name()
        event_filter = None
        client.call(
            "checkpoint",
            str(tmpdir),
            filename,
            event_filter,
        )
        client.close()
    except Exception as e:
        print(e)
    finally:
        lock.release()
        server.join()
    
    assert get_task(maybe_none=True) is None
    loaded_checkpoint = load_checkpoint(str(tmpdir), filename)
    saved, _, _ = loaded_checkpoint
    assert len(saved) == 2
    thread = next(iter(saved.values()))
    assert isinstance(thread, TaskThread)
    find_frame = False
    lock_type = type(Lock())
    for frame in thread.frames:
        assert isinstance(frame, FunctionFrame)
        if "acquire" in frame.states.func.__name__:
            for local_var in frame.states.nlocals:
                if isinstance(local_var, lock_type):
                    assert local_var.locked()
                    local_var.release()
                    find_frame = True
                    break
            if find_frame:
                break
    assert find_frame

    resume_checkpoint(loaded_checkpoint, f"localhost:{find_free_port()}")
