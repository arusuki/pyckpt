import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import FrameType, FunctionType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import dill

from pyckpt import interpreter, objects
from pyckpt.interpreter import ExceptionStates, snapshot_generator
from pyckpt.objects import SpawnContextManager

Analyzer = Callable[[FunctionType, int, bool], int]


class LiveFrame(ABC):
    def __init__(self):
        super().__init__()
        self._evaluated = False

    @abstractmethod
    def _evaluate(self): ...

    @abstractmethod
    def _cleanup(self): ...

    def evaluate(self, ret_val: Any = None, exc_states: Any = None):
        if self._evaluated:
            raise ValueError(f"frame {self} already evaluated")
        self._evaluated = True
        ret = self._evaluate(ret_val, exc_states)
        self._cleanup()
        return ret


class LiveFunctionFrame(LiveFrame):
    def __init__(
        self,
        *,
        func: FunctionType,
        prev_instr_offset: int,
        is_leaf: bool,
        nlocals: List[Any],
        stack: List[Any],
    ):
        super().__init__()
        self.is_leaf = is_leaf
        self.func = func
        self.stack = stack
        self.nlocals = nlocals
        self.prev_instr_offset = prev_instr_offset

    def _evaluate(
        self, ret_val, exc_states: Optional[ExceptionStates]
    ) -> Tuple[Any, Optional[ExceptionStates]]:
        ret = None
        ret = interpreter.eval_frame_at_lasti(
            self.func,
            self.nlocals,
            self.stack,
            self.is_leaf,
            ret_val,
            self.prev_instr_offset,
            exc_states,
        )
        return ret

    def _cleanup(self):
        del self.is_leaf
        del self.func
        del self.prev_instr_offset
        del self.stack
        del self.nlocals


class LiveGeneratorFrame(LiveFrame):
    def __init__(self, generator: Generator, is_leaf: bool):
        super().__init__()
        self._gen = generator
        self._is_leaf = is_leaf

    def _evaluate(
        self, ret_val, exc_states: Optional[ExceptionStates]
    ) -> Tuple[Any, Optional[ExceptionStates]]:
        return interpreter.resume_generator(
            self._gen, self._is_leaf, ret_val, exc_states
        )

    def _cleanup(self):
        del self._gen
        del self._is_leaf


@dataclass(frozen=True)
class FrameCocoon:
    is_leaf: bool
    func: Callable
    stack: bytes
    nlocals: bytes
    prev_instr_offset: int
    generator: Optional[Dict]

    @staticmethod
    def snapshot_from_frame(
        frame: FrameType,
        is_leaf: bool,
        stack_analyzer: Analyzer,
        contexts: Dict,
    ):
        captured = interpreter.snapshot(frame, is_leaf, stack_analyzer)
        nlocals = objects.snapshot_objects(captured["nlocals"], contexts)
        stack = objects.snapshot_objects(captured["stack"], contexts)
        generator = captured["generator"]
        if generator is not None:
            generator = snapshot_generator(generator)
        return FrameCocoon(
            is_leaf=is_leaf,
            func=captured["func"],
            nlocals=nlocals,
            stack=stack,
            prev_instr_offset=captured["prev_instr_offset"],
            generator=generator,
        )

    def _spawn_generator(self, nlocals: List, stack: List) -> LiveGeneratorFrame:
        gen = interpreter.make_generator(
            self.generator,
            {
                "func": self.func,
                "prev_instr_offset": self.prev_instr_offset,
                "nlocals": nlocals,
                "stack": stack,
            },
        )
        return LiveGeneratorFrame(gen, self.is_leaf)

    def spawn(self, contexts: SpawnContextManager) -> LiveFrame:
        nlocals = objects.spawn_objects(self.nlocals, contexts)
        stack = objects.spawn_objects(self.stack, contexts)
        if self.generator is None:
            return LiveFunctionFrame(
                func=self.func,
                is_leaf=self.is_leaf,
                stack=stack,
                nlocals=nlocals,
                prev_instr_offset=self.prev_instr_offset,
            )
        return self._spawn_generator(nlocals, stack)

    def clone(self) -> "FrameCocoon":
        return dill.copy(self)


def capture(backtrace_level=0):
    original_frame = inspect.currentframe()
    current = original_frame

    for _ in range(backtrace_level + 1):  # skip current frame
        current = current.f_back
        if current is None:
            raise ValueError(
                f"invalid backtrace level{backtrace_level}with frame Object {original_frame}"
            )
    return current
