import dis
import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from types import FrameType, FunctionType
from typing import Any, Callable, Generator, List, NamedTuple, Optional, Type

from pyckpt import interpreter
from pyckpt.analyzer import analyze_stack_size
from pyckpt.interpreter import EvaluateResult, ExceptionStates, get_generator

Analyzer = Callable[[FunctionType, int, bool], int]

CACHE = dis.opmap["CACHE"]
PRECALL = dis.opmap["PRECALL"]

logger = logging.getLogger(__name__)


CALL_INSTR_NAMES = ["CALL", "CALL_FUNCTION_EX"]
CALL_CODES = [dis.opmap[name] for name in CALL_INSTR_NAMES]

RETURN_INSTR_NAMES = ["RETURN_VALUE", "YIELD_VALUE"]
RETURN_CODES = [dis.opmap[name] for name in RETURN_INSTR_NAMES]


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
    def __init__(self, event: CaptureEvent, instr_offset: int):
        self.event = event
        self.instr_offset = instr_offset
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
        if self.event != CaptureEvent.INTERMEDIATE:
            logger.debug("evaluate: set instr offset to %s", self.instr_offset)
            self._set_instr_offset(self.instr_offset)
        return self._evaluate(exc_states)


class FunctionFrame(Frame):
    def __init__(
        self,
        event: CaptureEvent,
        instr_offset: int,
        states: Optional[FrameStates] = None,
    ):
        super().__init__(event, instr_offset)
        self.states = states
        self._override_instr_offset: Optional[int] = None

    def __str__(self):
        return "FunctionFrame: " + str(self.states)

    def _set_instr_offset(self, instr_offset: int):
        if not self.states:
            raise ValueError("invliad fuction frame")
        self._override_instr_offset = instr_offset

    def _set_return_value(self, value: Any):
        if not self.states:
            raise ValueError("invliad fuction frame")
        self.states.stack.append(value)

    def _evaluate(self, exc_states: ExceptionStates) -> EvaluateResult:
        logger.debug("evaluate: %s, override instr_offset: %s", self.states, self._override_instr_offset)
        if not self.states:
            raise ValueError("invliad fuction frame")
        instr_offset = (
            self.states.instr_offset
            if self._override_instr_offset is None
            else self._override_instr_offset
        )
        result = interpreter.eval_frame(
            func=self.states.func,
            nlocals=self.states.nlocals,
            stack=self.states.stack,
            instr_offset=instr_offset,
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
        instr_offset: int,
        generator: Optional[Generator],
        stack_size: int,
    ):
        super().__init__(event, instr_offset)
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

    def check_opcode(condition: Callable[[int], bool]):
        try:
            if instr_offset < 0:
                raise ValueError(f"invalid instr_offset(-1) for event {event}")
            opcode = code_array[instr_offset].opcode
            if not condition(opcode):
                op_name = dis.opname[opcode]
                raise ValueError(f"invalid opcode {op_name} for event {event}")
        except Exception:
            logger.fatal("frame last_i: %s, event: %s", frame.f_lasti, event)
            dis.dis(frame.f_code)
            raise

    def check_stack_size_with_args() -> tuple[int, int]:
        current_offset = instr_offset - 1
        redo_instr_offset = current_offset # CALL <- redo 
        current_offset = _fix_non_leaf_call(code_array, current_offset)

        if current_offset >= 0 and code_array[current_offset].opcode == PRECALL:
            current_offset -= 1
            redo_instr_offset = current_offset # PRECALL <- redo 
            current_offset = _fix_non_leaf_call(code_array, current_offset)
        return analyze_stack_size(
            code=frame.f_code,
            last_instr=_offset_without_cache(code_array, current_offset),
            is_generator=bool(generator),
        ), redo_instr_offset

    if event == CaptureEvent.INTERMEDIATE:
        check_opcode(lambda opcode: opcode in CALL_CODES)
        # remove the 'return value'
        stack_size -= 1

    elif event == CaptureEvent.CALL:
        if frame.f_lasti != 0:
            raise RuntimeError(
                f"frame {frame} call event with invalid frame last_i: {frame.f_lasti}"
            )
        instr_offset -= 1

    elif event == CaptureEvent.RETURN:
        check_opcode(lambda opcode: opcode in RETURN_CODES)
        # add one for return value
        if not generator:
            stack_size += 1
        # redo the return instruction
        instr_offset -= 1

    elif event == CaptureEvent.C_CALL:
        check_opcode(lambda opcode: opcode in CALL_CODES)
        stack_size, instr_offset = check_stack_size_with_args()

    elif event == CaptureEvent.C_RETURN or event == CaptureEvent.C_EXCEPTION:
        raise RuntimeError(f"receive unsupported event: {event}")

    else:
        logger.fatal("frame last_i: %s, event: %s", frame.f_lasti, event)
        dis.dis(frame.f_code)
        raise NotImplementedError(f"capture event not implemented: {event}")

    if generator:
        return GeneratorFrame(event, instr_offset, generator, stack_size)

    captured = interpreter.snapshot_frame(frame, stack_size)
    logger.debug("captured: %s", captured)

    return FunctionFrame(event, instr_offset, FrameStates.from_captured(captured))
