# pylint: disable=C0104,W0212,R0903
import dis
import inspect
from types import FrameType
from typing import Generator, Optional

import pytest
from bytecode import Bytecode, Instr

from pyckpt.interpreter import NullObject, EvaluateResult
from pyckpt.interpreter import frame as _frame


def test_eval_no_args():
    def foo():
        print("hello, world")

    _frame.eval_frame_at_lasti(
        foo,
        [],
        [],
        True,
    )


def test_eval_one_arg():
    def foo(arg1):
        print(f"hello: {arg1}")

    _frame.eval_frame_at_lasti(
        foo,
        ["world"],
        [],
        True,
    )


def test_eval_return():
    def foo():
        return "41"

    ins = dis.get_instructions(foo, show_caches=True)
    offset: Optional[int] = None
    RETURN_VALUE = dis.opmap["RETURN_VALUE"]
    for off, i in enumerate(ins):
        if i.opcode == RETURN_VALUE:
            offset = off
    assert offset
    ret, exc = _frame.eval_frame_at_lasti(
        foo, [], ["42"], True, is_return=True, prev_instr_offset=offset - 1
    )
    assert exc is None
    assert ret == "42"


def test_raise_exception():
    exc = RuntimeError("test raise exception")

    def test():
        raise exc

    ret, exc_states = _frame.eval_frame_at_lasti(
        test,
        [],
        [],
        True,
    )
    assert ret is NullObject
    assert isinstance(exc_states, tuple)
    assert exc_states[1] is exc


def test_return_add():
    def add(lhs, rhs):
        return lhs + rhs

    result, exc = _frame.eval_frame_at_lasti(
        add,
        [1, 2],
        [],
        True,
    )
    assert exc is None
    assert isinstance(result, int) and result == 3, "invalid result"


class FixedStackAnalyzer:
    def __init__(self, stack_size: int):
        self.stack_size = stack_size

    def __call__(self, *_unused_callable):
        return self.stack_size


def test_snapshot_stack():
    def print_and_return_last(*args):
        print(*args)
        return args[-1]

    def foo():
        frame = inspect.currentframe()
        arg1 = "hello"
        arg2 = "world"
        stack = print_and_return_last(
            arg1,
            arg2,
            _frame.snapshot(frame, True, FixedStackAnalyzer(4))["stack"],
        )

        assert stack[0] is NullObject
        assert stack[1] is print_and_return_last
        assert stack[2] is arg1
        assert stack[3] is arg2

    foo()


def test_snapshot_generator_from_frame():
    def foo():
        frame = inspect.currentframe()
        yield frame

    gen = foo()
    assert isinstance(gen, Generator)

    frame = next(gen)
    assert isinstance(frame, FrameType)
    g = _frame.get_generator(frame)
    assert isinstance(g, Generator)
    assert gen is g


def test_offset_without_cache():
    CACHE = dis.opmap["CACHE"]
    bytecode = Bytecode(
        [
            # <- offset=-1
            Instr("LOAD_GLOBAL", (True, "print")),  # <- offset= 0
            Instr("LOAD_CONST", "hello world"),  # <- offset= 1
            Instr("PRECALL", 1),  # <- offset= 2
            Instr("CALL", 1),  # <- offset= 3
            Instr("RETURN_VALUE"),  # <- offset= 4
        ]
    )
    code = bytecode.to_code()
    code_array = list(dis.get_instructions(code, show_caches=True))
    instr_counter = 0

    for idx, instr in enumerate(dis.get_instructions(code, show_caches=True)):
        if instr.opcode != CACHE:
            assert instr_counter == _frame._offset_without_cache(code_array, idx), (
                "invalid offset without cache"
            )

            instr_counter += 1


def test_reraise():
    e = RuntimeError("test")
    with pytest.raises(RuntimeError, match="test"):
        _frame.restore_thread_state({"exception": e})
        raise
    _frame.restore_thread_state({"exception": None})

def test_eval_traceback():
    
    def foo():
        raise RuntimeError("hello")

    def bar():
        pass

    eval_result = _frame.eval_frame_at_lasti(foo, [], [], True)
    assert isinstance(eval_result, EvaluateResult)
    ret, exc = eval_result
    assert ret is NullObject
    assert isinstance(exc, tuple)

    eval_result = _frame.eval_frame_at_lasti(bar, [], [], True, exc_states=exc)
    assert isinstance(eval_result, EvaluateResult)

    print(eval_result[1])
