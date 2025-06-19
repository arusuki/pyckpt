from dataclasses import dataclass
from typing import List, Dict, Optional, Type
from multiprocessing import Process
import os

from pyckpt.thread import LiveThread, ThreadCocoon
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

    def spawn(self, handle: Process) -> LiveProcess:
        live_process = LiveProcess(handle=handle, threads=self.threads)
        return live_process

    def clone(
        self, object_table: Dict, persist_mapping: Optional[Dict[Type, Mapping]] = None
    ) -> "ProcessCocoon":
        def persist_thread(p: Process):
            pid = id(p)
            if pid not in object_table:
                object_table[pid] = object.__new__(Process)
            return pid

        pm = persist_mapping if persist_mapping else {}
        pm.update({Process: persist_thread})
        if self.process_id not in object_table:
            object_table[self.process_id] = object.__new__(Process)
        return objects.copy(self, objects=object_table, persist_mapping=pm)
