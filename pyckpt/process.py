from dataclasses import dataclass
from typing import List, Dict, Any
from multiprocessing import Process
import os

from pyckpt.thread import LiveThread, ThreadCocoon


class LiveProcess:
    def __init__(
        self, handle: Process, threads: List[ThreadCocoon], exception_states: Dict
    ):
        self._resumed = False
        self._threads = threads
        self._handle = handle
        self._states = exception_states

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
    exception_states: Any

    def spawn(self):
        pass

    def clone(self) -> "ProcessCocoon":
        pass
