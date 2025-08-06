import json
import logging
from types import CodeType, FunctionType
from typing import Callable, Dict, List, Optional, Tuple

from bytecode import BasicBlock, Bytecode, ControlFlowGraph, SetLineno
from bytecode.instr import Instr, TryBegin, TryEnd

from pyckpt.util import dump_code_and_offset

Analyzer = Callable[[FunctionType, int, bool], int]

logger = logging.getLogger(__name__)


def _instruction_count(block: BasicBlock):
    return sum(1 for i in block if isinstance(i, Instr))


def _eval_offsets(cfg: ControlFlowGraph):
    result = [0]

    for block in cfg:
        result.append(_instruction_count(block) + result[-1])

    return result


SPECIAL_BLOCK_FLAG = -1

FINAL_RERAISE_BLOCK = -1


def _visit_block(
    visit_stack: List[BasicBlock],
    seen: Dict[int, int],
    next_block: BasicBlock,
    stack_size_after: int,
):
    assert isinstance(next_block, BasicBlock)
    assert stack_size_after >= 0

    if id(next_block) not in seen:
        seen[id(next_block)] = stack_size_after
        visit_stack.append((next_block, stack_size_after))
        return

    stack_size_previous = seen[id(next_block)]
    if stack_size_previous in (stack_size_after, SPECIAL_BLOCK_FLAG):
        return

    raise RuntimeError("different stack size when entering the same block")


def _symbolic_eval(cfg: ControlFlowGraph, is_generator: bool) -> Dict[int, int]:
    result: Dict[int, int] = {}
    seen: Dict[int, Tuple[int, int]] = {}
    visit_stack = []
    offsets = _eval_offsets(cfg)
    current: BasicBlock = next(iter(cfg))
    stack_size = 1 if is_generator else 0
    visit_stack.append((current, stack_size))
    seen[id(current)] = stack_size
    try_target: Optional[BasicBlock] = None

    result[-1] = 0
    while len(visit_stack) > 0:
        current, stack_size = visit_stack.pop()
        if seen[id(current)] == SPECIAL_BLOCK_FLAG:
            continue
        instr_index_base = offsets[cfg.get_block_index(current)]
        instr_idx = 0
        last_instr: Optional[Instr] = None

        for _, instr in enumerate(current):
            if isinstance(instr, Instr):
                last_instr = instr
                if instr.has_jump():
                    next_block = instr.arg
                    stack_size_after = stack_size + instr.stack_effect(True)
                    _visit_block(visit_stack, seen, next_block, stack_size_after)
                stack_size += instr.stack_effect(False)
                assert stack_size >= 0
                result[instr_index_base + instr_idx] = stack_size
                instr_idx += 1
            elif isinstance(instr, TryBegin):
                try_target = instr.target
                stack_size_after = instr.stack_depth + (2 if instr.push_lasti else 1)
                _visit_block(visit_stack, seen, try_target, stack_size_after)
            elif isinstance(instr, TryEnd):
                try_target = None
            else:
                assert isinstance(instr, SetLineno)

        if last_instr.is_final():
            continue

        next_block = current.next_block
        if next_block is not None:
            _visit_block(visit_stack, seen, next_block, stack_size)  # fallthrough

    return result


def analyze_stack_top(
    func: FunctionType,
    last_instr: int,
    is_generator: bool,
):
    code = Bytecode.from_code(func.__code__, conserve_exception_block_stackdepth=True)

    cfg = ControlFlowGraph.from_bytecode(code)
    try:
        eval_result = _symbolic_eval(cfg, is_generator)
    except RuntimeError as e:
        dump_code_and_offset(
            logger,
            func.__code__,
            last_instr,
            str(e),
            False,
        )
        raise

    if last_instr not in eval_result:
        error_msg = f"invalid instr idx:{last_instr},\
            analyze result:{json.dumps(eval_result, indent=4, ensure_ascii=False)}"
        dump_code_and_offset(
            logger,
            func.__code__,
            last_instr,
            error_msg,
            False,
        )
        raise RuntimeError(error_msg)

    return eval_result[last_instr]

def analyze_stack_size(
    code: CodeType,
    last_instr: int,
    is_generator: bool,
):
    byte_code = Bytecode.from_code(code, conserve_exception_block_stackdepth=True)

    cfg = ControlFlowGraph.from_bytecode(byte_code)
    try:
        eval_result = _symbolic_eval(cfg, is_generator)
    except RuntimeError as e:
        dump_code_and_offset(
            logger,
            code,
            last_instr,
            str(e),
            False,
        )
        raise

    if last_instr not in eval_result:
        error_msg = f"invalid instr idx:{last_instr},\
            analyze result:{json.dumps(eval_result, indent=4, ensure_ascii=False)}"
        dump_code_and_offset(
            logger,
            code,
            last_instr,
            error_msg,
            False,
        )
        raise RuntimeError(error_msg)

    return eval_result[last_instr]
