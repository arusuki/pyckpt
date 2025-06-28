import logging
from multiprocessing import Process
from threading import Thread
from typing import Optional
from io import BytesIO
from time import sleep
import multiprocessing
import threading
import traceback

from pyckpt import configure_logging
from pyckpt.process import (
    ProcessCocoon,
    snapshot_from_process,
    ProcessId,
)

configure_logging(logging.DEBUG)


def foo():
    sleep(1)


def bar():
    sleep(1)


def process_capture(q: multiprocessing.Queue, s: str):
    stream: Optional[bytes] = None
    process_id: Optional[ProcessId] = None
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
            process_id = process_cocoon.dump(file=buffer)
            stream = buffer.getvalue()
        q.put(process_cocoon)

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
    if isinstance(ret, Exception):
        raise RuntimeError("exception in capture process") from ret
    assert isinstance(ret, ProcessCocoon)
    assert len(ret.threads) == 2
    # assert isinstance(stream, bytes)
    # assert isinstance(process_id, ProcessId)

    # buffer = BytesIO()
    # buffer.write(stream)
    # buffer.seek(0)
    # c = ProcessCocoon.load(buffer, process_id)
    # objs = process_id()

    # assert isinstance(c, ProcessCocoon)
    # assert isinstance(objs, Dict)

    # assert id(p) in objs
    # live_process: LiveProcess = c.spawn(objs[id(p)])
    # live_process.evaluate(timeout=1.0)
    # result = capsys.readouterr()
    # assert result.out.count(s) == 1

