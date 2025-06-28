from dataclasses import dataclass
import logging
from types import CodeType
from typing import Generic, Optional, Tuple, TypeVar

from bytecode import Bytecode, ControlFlowGraph, format_bytecode, Instr


def format_instr(instr: Instr):
    return f"{instr.name}, {instr.arg} {instr.location}"

T = TypeVar("T")

E = TypeVar("E")

Result = Optional[Tuple[Optional[T], Optional[E]]]
NotNullResult = Tuple[Optional[T], Optional[E]]

@dataclass
class CodePosition:
    code: CodeType 
    offset: int
    reason: str

class BytecodeParseError(Exception):
    def __init__(self, pos: CodePosition):
        self._pos: CodePosition = pos

    def consume(self) -> CodePosition:
        pos = self._pos
        self.pos = None
        return pos

    def pos(self) -> CodePosition:
        return self._pos

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

def dump_code_position(logger: logging.Logger, pos: CodePosition, show_caches: bool):
    return dump_code_and_offset(
        logger, pos.code, pos.offset, pos.reason, show_caches
    )
