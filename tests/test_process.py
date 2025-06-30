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
from pyckpt.process import (ProcessCocoon, snapshot_from_process)

configure_logging(logging.DEBUG)


def foo():
    sleep(1)


def bar():
    sleep(1)


def process_capture(q: multiprocessing.Queue, s: str):
    byte_stream: Optional[bytes] = None

    try:
        t1 = Thread(target=foo)
        t2 = Thread(target=bar)
        t1.start()
        t2.start()
        process_cocoon, err = snapshot_from_process(multiprocessing.current_process())
        if err:
            raise err
        if process_cocoon is not None:
            buffer = BytesIO()
            objs = objects.dump(buffer, process_cocoon)
            byte_stream = buffer.getvalue()
        if not isinstance(q, FakeQueue):
            q.put((objs, byte_stream))

        t1.join()
        t2.join()

        print("executed")
    except Exception as e:
        traceback.print_exception(e)
        q.put(e)



def test_process_capture():
    q = multiprocessing.Queue()
    s = "hello_world"

    p = Process(target=process_capture, args=(q, s))
    p.start()
    ret = q.get(timeout=3.0)
    p.join()

    assert isinstance(ret, tuple) and len(ret) == 2
    objs, byte_stream = ret
    assert isinstance(objs, dict)
    assert isinstance(byte_stream, bytes)

    assert len(objs) == 2

    buffer = BytesIO(byte_stream)
    res = objects.load(buffer, objs)
    assert isinstance(res, tuple) and len(res) == 2
    c, objs = res
    assert isinstance(c, ProcessCocoon)
    assert isinstance(objs, dict)
    assert id(p) in objs

    
    # TODO: add evaluate support, current implementation is wrong.
    # live_process: LiveProcess = c.spawn(objs[id(p)])
    # live_process.evaluate(timeout=1.0)
    # result = capsys.readouterr()
    # assert result.out.count(s) == 1

