from abc import ABC, abstractmethod
from dataclasses import dataclass
import dis
from enum import IntEnum
from types import FrameType, FunctionType
from typing import Any, Callable, Generator, List, NamedTuple, Optional, Tuple, Type

from pyckpt import interpreter
from pyckpt.analyzer import analyze_stack_size
from pyckpt.interpreter import EvaluateResult, ExceptionStates, get_generator

Analyzer = Callable[[FunctionType, int, bool], int]

CACHE = dis.opmap["CACHE"]


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


@dataclass(frozen=True)
class GeneratorFrameCocoon:
    gen: Generator
    is_leaf: bool

    def spawn(self):
        return LiveGeneratorFrame(
            self.gen,
            self.is_leaf,
        )


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


CALL_INSTR_NAMES = ["CALL", "CALL_FUNCTION_EX"]
CALL_CODES = [dis.opmap[name] for name in CALL_INSTR_NAMES]


class CaptureEvent(IntEnum):
    INTERMEDIATE = -1
    CALL = 0
    EXCEPTION = 1
    LINE = 2
    RETURN = 3
    C_CALL = 4
    C_EXCEPTION = 5
    C_RETURN = 6
    OPCODE = 7


class FrameStates(NamedTuple):
    func: Callable
    stack: List[Any]
    nlocals: List[Any]
    instr_offset: int

    @classmethod
    def from_captured(cls: Type["FrameStates"], captured: dict):
        return cls(
            func=captured["func"],
            stack=captured["stack"],
            nlocals=captured["nlocals"],
            instr_offset=captured["instr_offset"],
        )


class Frame(ABC):
    def __init__(self, event: CaptureEvent):
        self.event = event
        self._requires_ret = event == CaptureEvent.INTERMEDIATE

    @abstractmethod
    def _set_instr_offset(self, instr_offset: int):
        pass

    @abstractmethod
    def _set_return_value(self, value: Any):
        pass

    @abstractmethod
    def _evaluate(self, exc_states: ExceptionStates) -> EvaluateResult:
        pass

    def return_value(self, value: Any):
        if not self._requires_ret:
            raise ValueError(f"{self} does not require return value")
        self._set_return_value(value)
        self._requires_ret = False
        return self

    def evaluate(self, exc_states: Optional[ExceptionStates] = None) -> EvaluateResult:
        if self._requires_ret:
            raise ValueError(
                f"{self} needs a return value, use `return_value()` to set a return value"
            )
        return self._evaluate(exc_states)


class FunctionFrame(Frame):
    def __init__(
        self,
        event: CaptureEvent,
        states: Optional[FrameStates] = None,
    ):
        super().__init__(event)
        self.states = states

    def _set_instr_offset(self, instr_offset: int):
        if not self.states:
            raise ValueError("invliad fuction frame")
        self.states.instr_offset = instr_offset

    def _set_return_value(self, value: Any):
        if not self.states:
            raise ValueError("invliad fuction frame")
        self.states.stack.append(value)

    def _evaluate(self, exc_states: ExceptionStates) -> EvaluateResult:
        if not self.states:
            raise ValueError("invliad fuction frame")
        result = interpreter.eval_frame(
            func=self.states.func,
            nlocals=self.states.nlocals,
            stack=self.states.stack,
            instr_offset=self.states.instr_offset,
            exc_states=exc_states,
        )
        self.states = None
        return result

class NotSetType: ...

NOT_SET = NotSetType()

class GeneratorFrame(Frame):

    def __init__(
        self,
        event: CaptureEvent,
        generator: Optional[Generator],
        stack_size: int,
    ):
        super().__init__(event)
        self.generator = generator
        self.stack_size = stack_size
        self._return_value = NOT_SET

    def _set_instr_offset(self, instr_offset: int):
        if not self.generator:
            raise ValueError("invliad generator frame")
        interpreter.generator_set_instr_offset(self.generator, instr_offset)

    def _set_return_value(self, value: Any):
        if not self.generator:
            raise ValueError("invliad generator frame")
        self._return_value = value

    def _evaluate(self, exc_states: ExceptionStates) -> EvaluateResult:
        if not self.generator:
            raise ValueError("invliad generator frame")
        if self.event != CaptureEvent.INTERMEDIATE:
            raise NotImplementedError("evaluate non-intermediate generator frame")
        interpreter.generator_shrink_stack(self.generator, self.stack_size)
        if self._return_value is not NOT_SET:
            interpreter.generator_push_stack(self.generator, self._return_value)
        result = interpreter.generator_resume(self.generator, exc_states)
        self.generator = None
        return result


def _offset_without_cache(
    code_array: List[dis.Instruction],
    offset_with_cache: int,
) -> int:
    if offset_with_cache < 0:
        return offset_with_cache
    return sum(1 for c in code_array[:offset_with_cache] if c.opcode != CACHE)


def _fix_non_leaf_call(code_array: List[dis.Instruction], instr_offset):
    """
    we need to manually move instr_offset back to the non-CACHE instruction
    """
    if instr_offset == -1:
        return -1
    instr = code_array[instr_offset]
    if instr.opcode == CACHE:
        current = instr_offset - 1
        while code_array[current].opcode == CACHE:
            current -= 1
        instr_offset = current
    return instr_offset


def snapshot_frame(
    frame: FrameType, event: CaptureEvent = CaptureEvent.INTERMEDIATE
) -> FunctionFrame | GeneratorFrame:
    generator = get_generator(frame)
    code_array = list(dis.get_instructions(frame.f_code, show_caches=True))
    instr_offset = _fix_non_leaf_call(code_array, frame.f_lasti // 2)
    stack_size = analyze_stack_size(
        code=frame.f_code,
        last_instr=_offset_without_cache(code_array, instr_offset),
        is_generator=bool(generator),
    )
    if event == CaptureEvent.INTERMEDIATE:
        if instr_offset < 0:
            raise ValueError("invalid instr_offset(-1) for intermediate frame")
        opcode = code_array[instr_offset].opcode
        if opcode not in CALL_CODES:
            op_name = dis.opname[opcode]
            raise ValueError(f"invalid opcode {op_name} for intermediate frame")
        # remove the 'return value'
        stack_size -= 1
    else:
        raise NotImplementedError(f"event: {event}")

    if generator:
        return GeneratorFrame(event, generator, stack_size)

    captured = interpreter.snapshot_frame(frame, stack_size)

    return FunctionFrame(event, FrameStates.from_captured(captured))
