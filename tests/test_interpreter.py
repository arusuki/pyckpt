# pylint: disable=C0104,W0212,R0903
import dis
import inspect
from typing import List

from bytecode import Bytecode, Instr
from pyckpt import interpreter
from pyckpt.frame import NullObject


def test_eval_no_args():

    def foo():
        print("hello, world")

    interpreter.eval_frame_at_lasti(
        foo,
        [],
        [],
    )


def test_eval_one_arg():

    def foo(arg1):
        print(f"hello: {arg1}")

    interpreter.eval_frame_at_lasti(
        foo,
        ["world"],
        [],
    )


def test_return_add():

    def add(lhs, rhs):
        return lhs + rhs

    result = interpreter.eval_frame_at_lasti(
        add,
        [1, 2],
        [],
    )

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
            interpreter.snapshot(frame, True, FixedStackAnalyzer(4))['stack']
        )

        assert stack[0] is NullObject
        assert stack[1] is print_and_return_last
        assert stack[2] is arg1
        assert stack[3] is arg2

    foo()


def _offset_without_cache(
    code_array: List[dis.Instruction],
    offset_with_cache: int,
) -> int:
    CACHE = dis.opmap['CACHE']

    if offset_with_cache < 0:
        return offset_with_cache

    return sum(1 for c in code_array[:offset_with_cache] if c.opcode != CACHE)


def test_offset_without_cache():

    CACHE = dis.opmap['CACHE']

    bytecode = Bytecode(
        [
            # <- offset=-1
            Instr('LOAD_GLOBAL', (True, 'print')),  # <- offset= 0
            Instr('LOAD_CONST', 'hello world'),     # <- offset= 1
            Instr('PRECALL', 1),                    # <- offset= 2
            Instr('CALL', 1),                       # <- offset= 3
            Instr('RETURN_VALUE'),                  # <- offset= 4
        ]
    )

    code = bytecode.to_code()
    code_array = list(dis.get_instructions(code, show_caches=True))
    instr_counter = 0

    for idx, instr in enumerate(
        dis.get_instructions(code, show_caches=True)
    ):
        if instr.opcode != CACHE:
            assert instr_counter == _offset_without_cache(code_array, idx), \
                "invalid offset without cache"

            instr_counter += 1
