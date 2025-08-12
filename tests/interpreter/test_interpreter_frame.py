import dis
import inspect
from types import CodeType, FrameType
from typing import Generator, Optional

import pytest
from bytecode import Bytecode, Instr

from pyckpt.interpreter import EvaluateResult, NullObject
from pyckpt.interpreter import frame as _frame


def test_eval_frame_no_args(capsys):
    def foo():
        print("hello")

    _frame.eval_frame(foo, [], [], -1)
    result = capsys.readouterr()
    assert result.out.count("hello") == 1


def test_eval_frame_one_arg(capsys):
    def foo(arg1):
        print(f"hello {arg1}")

    _frame.eval_frame(foo, ["world"], [], -1)
    result = capsys.readouterr()
    assert result.out.count("hello world") == 1

def test_eval_frame_return():
    def foo():
        return "41"

    ins = dis.get_instructions(foo, show_caches=True)
    offset: Optional[int] = None
    RETURN_VALUE = dis.opmap["RETURN_VALUE"]
    for off, i in enumerate(ins):
        if i.opcode == RETURN_VALUE:
            offset = off
    assert offset
    ret, exc = _frame.eval_frame(foo, [], ["42"], instr_offset=offset - 1)
    assert exc is None
    assert ret == "42"


def test_eval_frame_raise_exception():
    exc = RuntimeError("test raise exception")

    def test():
        raise exc

    ret, exc_states = _frame.eval_frame(
        test,
        [],
        [],
        -1,
    )
    assert ret is NullObject
    assert isinstance(exc_states, tuple)
    assert exc_states[1] is exc


def test_eval_frame_return_add():
    def add(lhs, rhs):
        return lhs + rhs

    result, exc = _frame.eval_frame(
        add,
        [1, 2],
        [],
        -1,
    )
    assert exc is None
    assert isinstance(result, int) and result == 3, "invalid result"


def test_eval_frame_traceback():
    def foo():
        raise RuntimeError("hello")

    def bar():
        pass

    eval_result = _frame.eval_frame(foo, [], [], -1)
    assert isinstance(eval_result, EvaluateResult)
    ret, exc = eval_result
    assert ret is NullObject
    assert isinstance(exc, tuple)

    eval_result = _frame.eval_frame(bar, [], [], -1, exc_states=exc)
    assert isinstance(eval_result, EvaluateResult)


def test_restore_thread_states():
    e = RuntimeError("test")
    with pytest.raises(RuntimeError, match="test"):
        _frame.restore_thread_state({"exception": e})
        raise
    _frame.restore_thread_state({"exception": None})


def test_snapshot_frame_stack():
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
            _frame.snapshot_frame(frame, 4)["stack"],
        )

        assert stack[0] is NullObject
        assert stack[1] is print_and_return_last
        assert stack[2] is arg1
        assert stack[3] is arg2

    foo()


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


def test_frame_lasti_opcode():

    def foo():
        bar()

    def bar():
        frame = inspect.currentframe().f_back
        opcode = _frame.frame_lasti_opcode(frame)
        assert opcode == dis.opmap["CALL"]

    foo()
