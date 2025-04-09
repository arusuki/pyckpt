from typing import List

from bytecode import (
    BasicBlock,
    Bytecode,
    ControlFlowGraph,
    Instr,
    TryBegin,
    dump_bytecode,
)

from pyckpt.analyzer import _symbolic_eval


def _get_active_blocks(cfg: ControlFlowGraph) -> List[BasicBlock]:
    if not cfg:
        return []

    seen_block_ids = set()
    stack = [cfg[0]]
    while stack:
        block = stack.pop()
        if id(block) in seen_block_ids:
            continue
        seen_block_ids.add(id(block))

        fall_through = True
        for i in block:
            if isinstance(i, Instr):
                if isinstance(i.arg, BasicBlock):
                    stack.append(i.arg)
                if i.is_final():
                    fall_through = False
            elif isinstance(i, TryBegin):
                assert isinstance(i.target, BasicBlock)
                stack.append(i.target)
        if fall_through and block.next_block:
            stack.append(block.next_block)

    return [b for b in cfg if id(b) in seen_block_ids]


def _get_active_instr_count(cfg: ControlFlowGraph):
    blocks = _get_active_blocks(cfg)
    cnt = 0
    for b in blocks:
        cnt += sum(1 for instr in b if isinstance(instr, Instr))
    return cnt


def test_push_pop():
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

    reference = {
        -1: 0,
        0: 2,
        1: 3,
        2: 2,
        3: 1,
        4: 0,
    }

    result = _symbolic_eval(ControlFlowGraph.from_bytecode(bytecode), False)

    for offset, size in result.items():
        assert offset in reference
        assert size == reference[offset]


def test_control_flow():
    def foo(condition: bool):
        if condition:
            print("yes")
        else:
            print("no")

        return not condition

    code = Bytecode.from_code(foo.__code__, conserve_exception_block_stackdepth=True)
    result = _symbolic_eval(ControlFlowGraph.from_bytecode(code), False)
    dump_bytecode(code, lineno=True)
    code_length = sum(1 for i in code if isinstance(i, Instr))
    assert len(result) == code_length + 1, "Analyzer not cover full code path"
    assert max(result.keys()) == code_length - 1
    assert min(result.keys()) == -1


def test_analyze_try_clause():
    def function_with_try():
        try:
            print("1")
        except RuntimeError:
            print("2")
        print("3")

    code = Bytecode.from_code(
        function_with_try.__code__, conserve_exception_block_stackdepth=True
    )
    cfg = ControlFlowGraph.from_bytecode(code)
    result = _symbolic_eval(cfg, False)
    code_length = sum(1 for i in code if isinstance(i, Instr))
    assert len(result) == code_length + 1, "Analyzer not cover full code path"
    assert max(result.keys()) == code_length - 1
    assert min(result.keys()) == -1


def test_analyze_try_in_for_loop():
    def func():
        for _ in range(10):
            try:
                print("1")
            except StopIteration:
                print("2")
                continue
            except RuntimeError:
                print("3")
                continue

            print("4")

    code = Bytecode.from_code(func.__code__, conserve_exception_block_stackdepth=True)
    cfg = ControlFlowGraph.from_bytecode(code)
    result = _symbolic_eval(cfg, False)
    code_length = sum(1 for i in code if isinstance(i, Instr))
    assert len(result) == code_length + 1, "Analyzer not cover full code path"
    assert max(result.keys()) == code_length - 1
    assert min(result.keys()) == -1


def test_analyze_nested_try_clause():
    def func():
        try:
            pass
        except RuntimeError:
            pass
        finally:
            for _ in range(10):
                try:
                    print("1")
                except StopIteration:
                    print("2")

    code = Bytecode.from_code(func.__code__, conserve_exception_block_stackdepth=True)
    cfg = ControlFlowGraph.from_bytecode(code)
    result = _symbolic_eval(cfg, False)
    code_length = _get_active_instr_count(cfg)
    print(result)
    assert len(result) == code_length + 1, "Analyzer not cover full code path"


def test_analyze_generator():
    def test():
        yield 1

    code = Bytecode.from_code(test.__code__, conserve_exception_block_stackdepth=True)
    cfg = ControlFlowGraph.from_bytecode(code)
    result = _symbolic_eval(cfg, True)
    code_length = _get_active_instr_count(cfg)
    print(result)
    assert len(result) == code_length + 1, "Analyzer not cover full code path"
