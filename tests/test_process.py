import logging
from multiprocessing import Process
from threading import Thread
from typing import Optional
from io import BytesIO
from time import sleep
import multiprocessing
import traceback


from pyckpt import configure_logging, objects

from pyckpt.objects import FakeQueue
from pyckpt.process import (LiveProcess, ProcessCocoon, snapshot_from_process)
from pyckpt.thread import mark_truncate

configure_logging(logging.DEBUG)


def foo():
    sleep(1)


def bar():
    sleep(1)


def process_capture(q: multiprocessing.Queue):
    byte_stream: Optional[bytes] = None

    try:
        t1 = Thread(target=foo)
        t2 = Thread(target=bar)
        t1.start()
        t2.start()
        ret = mark_truncate(snapshot_from_process, multiprocessing.current_process())
        if ret is None:
            print("spawned")
            return
        process_cocoon, err = ret
        if err:
            raise err
        if process_cocoon is not None:
            buffer = BytesIO()
            objs = process_cocoon.dump(buffer)
            byte_stream = buffer.getvalue()
        if not isinstance(q, FakeQueue):
            q.put((objs, byte_stream))

        t1.join()
        t2.join()

        print("executed")
    except Exception as e:
        traceback.print_exception(e)
        if not isinstance(q, FakeQueue):
            q.put(e)



def test_process_capture():
    q = multiprocessing.Queue()

    p = Process(target=process_capture, args=(q,))
    p.start()
    ret = q.get(timeout=3.0)
    p.join()

    assert isinstance(ret, tuple) and len(ret) == 2
    objs, byte_stream = ret
    assert isinstance(objs, dict)
    assert isinstance(byte_stream, bytes)
    assert len(objs[Thread]) == 3

    buffer = BytesIO(byte_stream)
    res = objects.load(buffer, objs)
    assert isinstance(res, tuple) and len(res) == 2
    c, pool = res
    assert isinstance(c, ProcessCocoon)

    assert isinstance(pool, dict)
    assert id(p) in pool

    live_process: LiveProcess = c.spawn(pool)
    assert live_process.evaluate(timeout=1.0) == 0

