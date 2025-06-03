from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import FrameType, FunctionType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type

from pyckpt import interpreter, objects
from pyckpt.interpreter import EvaluateResult, ExceptionStates, get_generator
from pyckpt.objects import Mapping

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
        if not isinstance(ret, EvaluateResult):
            ret = EvaluateResult(ret, None)
        self._cleanup()
        return ret


class LiveFunctionFrame(LiveFrame):
    def __init__(
        self,
        *,
        func: FunctionType,
        prev_instr_offset: int,
        is_leaf: bool,
        is_return: bool,
        nlocals: List[Any],
        stack: List[Any],
    ):
        super().__init__()
        self.is_leaf = is_leaf
        self.func = func
        self.stack = stack
        self.nlocals = nlocals
        self.prev_instr_offset = prev_instr_offset
        self.is_return = is_return

    def _evaluate(
        self, ret_val, exc_states: Optional[ExceptionStates]
    ) -> Tuple[Any, Optional[ExceptionStates]]:
        ret = None
        ret = interpreter.eval_frame_at_lasti(
            self.func,
            self.nlocals,
            self.stack,
            self.is_leaf,
            self.is_return,
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


@dataclass
class FunctionFrameCocoon:
    is_leaf: bool
    is_return: bool
    func: Callable
    stack: List[Any]
    nlocals: List[Any]
    prev_instr_offset: int

    def spawn(self) -> LiveFrame:
        return LiveFunctionFrame(
            func=self.func,
            is_leaf=self.is_leaf,
            is_return=self.is_return,
            stack=self.stack,
            nlocals=self.nlocals,
            prev_instr_offset=self.prev_instr_offset,
        )

    def clone(
        self,
        object_table: Optional[Dict] = None,
        persist_mapping: Optional[Dict[Type, Mapping]] = None,
    ) -> "FunctionFrameCocoon":
        return objects.copy(self, objects=object_table, persist_mapping=persist_mapping)


@dataclass(frozen=True)
class GeneratorFrameCocoon:
    gen: Generator
    is_leaf: bool

    def spawn(self):
        return LiveGeneratorFrame(
            self.gen,
            self.is_leaf,
        )

    def clone(self):
        object_table: Optional[Dict] = None
        persist_mapping: Optional[Dict[Type, Mapping]] = None
        return objects.copy(self, objects=object_table, persist_mapping=persist_mapping)


def snapshot_from_frame(
    frame: FrameType,
    is_leaf: bool,
    stack_analyzer: Analyzer,
):
    generator = get_generator(frame)
    if generator:
        return GeneratorFrameCocoon(generator, is_leaf)

    captured = interpreter.snapshot(frame, is_leaf, stack_analyzer)
    return FunctionFrameCocoon(
        is_leaf=is_leaf,
        is_return=captured["is_return"],
        func=captured["func"],
        nlocals=captured["nlocals"],
        stack=captured["stack"],
        prev_instr_offset=captured["prev_instr_offset"],
    )
