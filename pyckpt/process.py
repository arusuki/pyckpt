import os
from dataclasses import dataclass
from multiprocessing import Process
from threading import Thread
from types import FrameType
from typing import IO, Dict, List, Optional, Type, cast

import pyckpt.objects as objects
from pyckpt.objects import Mapping
from pyckpt.thread import (
    StarPlatinum,
    ThreadCocoon,
    ThreadId,
    snapshot_from_thread,
)
from pyckpt.util import NotNullResult


class ProcessId:
    def __init__(self, process_id, thread_ids: List[ThreadId]):
        self.pid = process_id
        self.tids = thread_ids

    def __call__(self):
        objs = {self.pid: object.__new__(Process)}
        for tid in self.tids:
            objs.update(tid())
        return objs


class LiveProcess:
    def __init__(self, handle: Process, threads: List[ThreadCocoon]):
        self._resumed = False
        self._threads = threads
        self._handle = handle

        handle.__init__(target=self._evaluate)

    def _evaluate(self):
        threads = [t.spawn() for t in self._threads]

        for t in threads:
            t.evaluate()

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
            raise RuntimeError("can't spawn a processcocoon before registering it")
        live_process = LiveProcess(handle=handle, threads=self.threads)
        return live_process

    def dump(
        self, file: IO[bytes], persist_mapping: Optional[Dict] = None
    ) -> ProcessId:
        process_ids = set()

        def persist_process(p: Process):
            pid = id(p)
            if pid not in process_ids:
                process_ids.add(pid)
            return pid

        pm = persist_mapping if persist_mapping else {}
        pm.update({Process: persist_process})
        pickler = objects.create_pickler(file, pm)

        tids = []
        for t in self.threads:
            print(self.threads)
            tids.append(t.dump(pickler=pickler))

        objects.dump(pickler, self)
        pid = ProcessId(self.process_id, tids)

        return pid

    @staticmethod
    def load(
        file: IO[bytes],
        process_id: ProcessId,
    ) -> "ProcessCocoon":
        objs = process_id()
        return objects.load(file, objs)

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

def extract_threads(
    running: Dict[Thread, FrameType], 
    waiting: Dict[Thread, FrameType]
) -> NotNullResult[List[ThreadCocoon], Exception]:
    all_threads = running | waiting
    threads: List[ThreadCocoon] = []
    for t, frame in all_threads.items():
        ret = snapshot_from_thread(t, frame=frame)
        if not ret:
            return (None, RuntimeError("recover at this function is not allowed"))
        thread_cocoon, err = ret
        if err:
            return (None, err)
        threads.append(thread_cocoon)
    return (threads, None)


def snapshot_from_process(p: Process, timeout=None) -> NotNullResult[ProcessCocoon, Exception]:
    assert p.pid == os.getpid()
    threads: Optional[List[ThreadCocoon]] = None
    err: Optional[Exception] = None

    def snapshot():
        nonlocal threads, err
        stopper = StarPlatinum(operation=extract_threads, timeout=timeout)
        threads, err = stopper.THE_WORLD()

    t = Thread(target=snapshot)
    t.start()
    t.join()
    if err or not threads:
        return (None, cast(Exception, err))


    return (ProcessCocoon(id(p), threads), None)

