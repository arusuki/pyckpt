from multiprocessing import Process
from threading import Thread
from typing import Optional, Dict, Callable
from io import BytesIO
from time import sleep
import multiprocessing
import threading

from pyckpt.process import (
    ProcessCocoon,
    snapshot_from_process,
    LiveProcess,
    ProcessId,
)


def foo():
    sleep(1)


def bar():
    sleep(1)


def process_capture(q: multiprocessing.Queue, s: str):
    stream: Optional[bytes] = None
    process_id: Optional[ProcessId] = None

    t1 = Thread(target=foo)
    t2 = Thread(target=bar)

    t1.start()
    t2.start()

    print(f"t1:{id(t1)}")
    print(f"t2:{id(t2)}")
    process_cocoon = snapshot_from_process(multiprocessing.current_process())
    if process_cocoon is not None:
        buffer = BytesIO()
        process_id = process_cocoon.dump(file=buffer)
        stream = buffer.getvalue()

    print(id(multiprocessing.current_process()))
    print("threading.enumerate():")
    for t in threading.enumerate():
        print(id(t))

    print(process_id().keys())

    q.put((stream, process_id))
    q.close()

    t1.join()
    t2.join()

    print(s)


def test_process_capture():
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        print("Start method has already been set.")

    q = multiprocessing.Queue()
    s = "hello_world"

    p = Process(target=process_capture, args=(q, s))
    p.start()
    stream, process_id = q.get(timeout=5.0)
    p.join()

    assert isinstance(stream, bytes)
    assert isinstance(process_id, ProcessId)

    buffer = BytesIO()
    buffer.write(stream)
    buffer.seek(0)
    c = ProcessCocoon.load(buffer, process_id)
    objs = process_id()

    # assert isinstance(c, ProcessCocoon)
    # assert isinstance(objs, Dict)

    # assert id(p) in objs
    # live_process: LiveProcess = c.spawn(objs[id(p)])
    # live_process.evaluate(timeout=1.0)
    # result = capsys.readouterr()
    # assert result.out.count(s) == 1


test_process_capture()
