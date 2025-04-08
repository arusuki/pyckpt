from dataclasses import dataclass
import inspect
from types import FrameType, FunctionType
from typing import Any, Callable, Dict, List, Type
from pyckpt import interpreter
from pyckpt.analyzer import Analyzer
from pyckpt import objects
from pyckpt.objects import HookType


class NullObjectType:
    pass


NullObject = NullObjectType()


@dataclass
class FrameStates:
    is_leaf: bool
    func: Callable
    stack: List[Any]
    prev_instr_offset: int
    nlocals: List[Any]


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
        registry: Dict[Type, HookType] = None,
    ):
        self._evaluated = False
        self.registry = registry

        self.states = FrameStates(
            is_leaf=is_leaf,
            func=func,
            stack=stack,
            nlocals=nlocals,
            prev_instr_offset=prev_instr_offset,
        )

    def __getattr__(self, name):
        return getattr(self.states, name)

    def __getstate__(self):
        if self._evaluated:
            raise RuntimeError("evaluate frame that is already evaluated.")

        if self.registry is None:
            raise RuntimeError(
                "Saved frame with None registry is not serializable")
        states = self.states.__dict__.copy()
        for attr in ('stack', 'nlocals'):
            states[attr] = objects.stub_objects(
                self.registry, getattr(self.states, attr))
            return states

    def __setstate__(self, original_states: Dict):
        states = original_states.copy()
        for attr in ('stack', 'nlocals'):
            states[attr] = objects.get_real_objects(states[attr])
        self.states = object.__new__(FrameStates)
        self.states.__dict__.update(states)
        self._evaluated = False
        self.registry = None

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
        del self.states.nlocals
        del self.states.stack

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
