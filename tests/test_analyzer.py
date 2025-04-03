from bytecode import Bytecode, ControlFlowGraph, Instr, Label, dump_bytecode

from pyckpt.analyzer import _symbolic_eval


def test_push_pop():

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

    reference = {
        -1: 0,
        0: 2,
        1: 3,
        2: 2,
        3: 1,
        4: 0,
    }

    result = _symbolic_eval(ControlFlowGraph.from_bytecode(bytecode))

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

    code = Bytecode.from_code(foo.__code__)
    result = _symbolic_eval(ControlFlowGraph.from_bytecode(code))
    dump_bytecode(code, lineno=True)
    code_length = sum(1 for i in code if not isinstance(i, Label))
    assert len(result) == code_length + 1, "Analyzer not cover full code path"
