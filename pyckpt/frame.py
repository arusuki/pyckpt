import inspect
from dataclasses import dataclass
from types import FrameType, FunctionType
from typing import Any, Callable, Dict, List, Optional, Tuple

import dill

from pyckpt import interpreter, objects
from pyckpt.interpreter import ExceptionStates

Analyzer = Callable[[FunctionType, int, bool], int]


class LiveFrame:
    def __init__(
        self,
        *,
        func: FunctionType,
        prev_instr_offset: int,
        is_leaf: bool,
        nlocals: List[Any],
        stack: List[Any],
    ):
        self._evaluated = False
        self.is_leaf = is_leaf
        self.func = func
        self.stack = stack
        self.nlocals = nlocals
        self.prev_instr_offset = prev_instr_offset

    def evaluate(
        self, ret_val=None, exc_states: Optional[ExceptionStates] = None
    ) -> Tuple[Any, Optional[ExceptionStates]]:
        ret = None
        if self._evaluated:
            raise RuntimeError("evaluate frame that is already evaluated.")
        ret = interpreter.eval_frame_at_lasti(
            self.func,
            self.nlocals,
            self.stack,
            self.is_leaf,
            ret_val,
            self.prev_instr_offset,
            exc_states,
        )
        # remove local variables
        del self.nlocals
        del self.stack
        self._evaluated = True
        return ret


@dataclass(frozen=True)
class FrameCocoon:
    is_leaf: bool
    func: Callable
    stack: bytes
    nlocals: bytes
    prev_instr_offset: int

    @staticmethod
    def from_frame(
        frame: FrameType,
        is_leaf: bool,
        stack_analyzer: Analyzer,
        stub_registry: Dict,
        contexts: Dict,
    ):
        captured = interpreter.snapshot(frame, is_leaf, stack_analyzer)
        nlocals = objects.stub_objects(stub_registry, captured["nlocals"], contexts)
        stack = objects.stub_objects(stub_registry, captured["stack"], contexts)
        captured["nlocals"] = nlocals
        captured["stack"] = stack
        if captured["generator"] is not None:
            raise NotImplementedError("snapshot generator is not implemented")
        del captured["generator"]
        return FrameCocoon(**captured)

    def spawn(self, contexts: Dict) -> LiveFrame:
        nlocals = objects.get_real_objects(self.nlocals, contexts)
        stack = objects.get_real_objects(self.stack, contexts)
        return LiveFrame(
            func=self.func,
            is_leaf=self.is_leaf,
            stack=stack,
            nlocals=nlocals,
            prev_instr_offset=self.prev_instr_offset,
        )

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
