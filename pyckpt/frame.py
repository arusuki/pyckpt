import inspect
from types import FrameType, FunctionType
from typing import Any, List
from pyckpt import interpreter
from pyckpt.analyzer import Analyzer


class NullObjectType:
    pass


NullObject = NullObjectType()


class SavedFrame:

    @staticmethod
    def snapshot_from_frame(
        frame: FrameType,
        is_leaf: bool,
        stack_analyzer: Analyzer,
    ):
        captured = interpreter.snapshot(frame, is_leaf, stack_analyzer)
        return SavedFrame(**captured)

    def __init__(
        self,
        func: FunctionType,
        nlocals: List[Any],
        stack: List[Any],
        prev_instr_offset: int,
        is_leaf: bool,
    ):
        self._evaluated = False

        self.func = func
        self.nlocals = nlocals
        self.stack = stack
        self.prev_instr_offset = prev_instr_offset
        self.is_leaf = is_leaf

    def evaluate(self, ret_val=None):

        if self._evaluated:
            raise RuntimeError("evaluate frame that is already evaluated.")

        ret = interpreter.eval_frame_at_lasti(
            self.func,
            self.nlocals,
            self.stack,
            self.is_leaf,
            ret_val,
            self.prev_instr_offset
        )

        # remove local variables
        del self.nlocals
        del self.stack

        self._evaluated = True

        return ret


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
