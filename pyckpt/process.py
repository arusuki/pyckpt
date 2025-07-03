from io import BytesIO
import logging
import os
from dataclasses import dataclass
from multiprocessing import Process
from threading import Thread
from types import FrameType
from typing import Any, Dict, List, Optional, cast

from pyckpt import objects
from pyckpt.objects import PersistedObjects
from pyckpt.thread import (
    StarPlatinum,
    ThreadCocoon,
    snapshot_from_thread,
)
from pyckpt.util import NotNullResult, Result

logger = logging.getLogger(__name__)

class LiveProcess:
    def __init__(
        self,
        process_id: int,
        object_pools: Dict[int, Any],
        threads: List[ThreadCocoon],
    ):
        self.process_id = process_id

        self._resumed = False
        self._threads = threads
        self._handle = object_pools[process_id]
        self._objects = object_pools
        self._handle.__init__(target=self._evaluate)

    def _evaluate(self):
        threads = [
            t.spawn(self._objects[t.thread_id]) for t in self._threads
        ]

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
        return self._handle.exitcode


@dataclass
class ProcessCocoon:
    process_id: int
    threads: List[ThreadCocoon]

    def spawn(self, object_pool: Dict[int, Any]) -> LiveProcess:
        live_process = LiveProcess(
            self.process_id,
            object_pool,
            self.threads,
        )
        return live_process

    def dump(self, file: BytesIO) -> PersistedObjects:
        persisted = objects.dump(file, self)
        persisted[Process].add(self.process_id)
        persisted[Thread].update(t.thread_id for t in self.threads)
        return persisted

def extract_threads(
    running: Dict[Thread, FrameType], 
    waiting: Dict[Thread, FrameType]
) -> NotNullResult[List[ThreadCocoon], Exception]:
    all_threads = running | waiting
    threads: List[ThreadCocoon] = []
    for t, frame in all_threads.items():
        ret = snapshot_from_thread(t, frame=frame, capture_other=True)
        if not ret:
            return (None, RuntimeError("recover at this function is not allowed"))
        thread_cocoon, err = ret
        if err:
            return (None, err)
        threads.append(thread_cocoon)
    return (threads, None)


def snapshot_from_process(p: Process, timeout=None) -> Result[ProcessCocoon, Exception]:
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

