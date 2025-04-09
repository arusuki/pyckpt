import logging
from types import CodeType

from bytecode import Bytecode, ControlFlowGraph, format_bytecode, Instr


def format_instr(instr: Instr):
    return f"{instr.name}, {instr.arg} {instr.location}"


def dump_code_and_offset(
    logger: logging.Logger,
    code: CodeType,
    offset: int,
    msg: str,
    show_caches: bool,
):
    logger.debug(f"dump code object: {code}")
    logger.debug(
        format_bytecode(
            ControlFlowGraph.from_bytecode(
                Bytecode.from_code(code, conserve_exception_block_stackdepth=True)
            ),
            lineno=True,
        )
    )
    logger.debug(msg)
    code_array = list(Bytecode.from_code(code, prune_caches=not show_caches))
    logger.debug(
        "CODE_ARRAY:\n"
        + "\n".join(
            f"{i}: {format_instr(instr)}"
            + ("   <---- offset ----   " if i == offset else "")
            for i, instr in enumerate(
                ins for ins in code_array if isinstance(ins, Instr)
            )
        )
    )
