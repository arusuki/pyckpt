from multiprocessing import Process
from threading import Thread
from typing import Optional, Dict
import time
import multiprocessing

from pyckpt.process import ProcessCocoon, snapshot_from_process, LiveProcess


def process_capture(q: multiprocessing.Queue, s: str):
    c: Optional[ProcessCocoon] = None
    objs: Optional[Dict] = {}

    def foo():
        time.sleep(1)

    def bar():
        time.sleep(1)

    t1 = Thread(target=foo)
    t2 = Thread(target=bar)

    t1.start()
    t2.start()

    process_cocoon = snapshot_from_process(multiprocessing.current_process())
    if process_cocoon is not None:
        c = process_cocoon.clone(objs)
    q.put((c, objs))

    t1.join()
    t2.join()

    print(s)


def test_process_capture(capsys):
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        print("Start method has already been set.")

    s = "hello_world"
    q = multiprocessing.Queue()

    p = Process(target=process_capture, args=(q, s))
    p.start()
    c, objs = q.get(timeout=10)
    p.join()
