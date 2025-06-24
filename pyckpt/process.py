from dataclasses import dataclass
from typing import List, Dict, Optional, Type
from types import FrameType
from multiprocessing import Process
from threading import Thread
import os

from pyckpt.thread import LiveThread, ThreadCocoon, snapshot_from_thread, StarPlatinum
import pyckpt.objects as objects
from pyckpt.objects import Mapping


class LiveProcess:
    def __init__(self, handle: Process, threads: List[ThreadCocoon]):
        self._resumed = False
        self._threads = threads
        self._handle = handle

        handle.__init__(target=self._evaluate)

    def _evaluate(self):
        threads = [thread.spawn() for thread in self._threads]

        for thread in threads:
            thread.evaluate()

    @property
    def handle(self) -> Process:
        return self._handle

    def evaluate(self, timeout=None):
        if self._resumed:
            raise RuntimeError("re-evaluating a process")
        self._resumed = True
        self._handle.start()
        self._handle.join(timeout)


@dataclass
class ProcessCocoon:
    process_id: int
    threads: List[ThreadCocoon]
    _registered: bool = False

    def spawn(self, handle: Process) -> LiveProcess:
        if not self._registered:
            raise RuntimeError("can't spawn a processcocoon before cloning it")
        live_process = LiveProcess(handle=handle, threads=self.threads)
        return live_process

    def clone(
        self, object_table: Dict, persist_mapping: Optional[Dict[Type, Mapping]] = None
    ) -> "ProcessCocoon":
        def persist_process(p: Process):
            pid = id(p)
            if pid not in object_table:
                object_table[pid] = object.__new__(Process)
            return pid

        self.threads = [
            c.clone(object_table=object_table, persist_mapping=persist_mapping)
            for c in self.threads
        ]
        self._registered = True
        pm = persist_mapping if persist_mapping else {}
        pm.update({Process: persist_process})
        if self.process_id not in object_table:
            object_table[self.process_id] = object.__new__(Process)
        return objects.copy(self, objects=object_table, persist_mapping=pm)


def extract_threads(running: Dict[Thread, FrameType], waiting: Dict[Thread, FrameType]):
    all = running | waiting
    threads: List[ThreadCocoon] = [
        snapshot_from_thread(t, frame=f) for t, f in all.items()
    ]

    return threads


def snapshot_from_process(p: Process, timeout=None):
    assert p.pid == os.getpid()

    stopper = StarPlatinum(operation=extract_threads, timeout=timeout)
    threads = stopper.THE_WORLD()

    return ProcessCocoon(id(p), threads)
