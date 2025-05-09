import sys
from queue import Queue
from threading import Thread

import pytest

from pyckpt import platform

if sys.platform != "darwin":
    pytest.skip("These tests only run on darwin platform", allow_module_level=True)


def test_native_handle():
    q = Queue()

    def hello():
        q.get()

    t = Thread(target=hello)
    t.start()
    assert platform.is_valid_running_thread(t)
    q.put(None)
    t.join()


def test_suspend_resume_thread():
    finished = False

    def hello():
        while not finished:
            print("hello world")

    t = Thread(target=hello)
    t.start()
    platform.suspend_thread(t)
    platform.resume_thread(t)
    finished = True
    t.join()
