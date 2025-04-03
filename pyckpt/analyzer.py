from types import FunctionType
from typing import Callable, Dict, Tuple
from bytecode import ControlFlowGraph, Bytecode, BasicBlock


Analyzer = Callable[[FunctionType, int, bool], int]


def _eval_offsets(cfg: ControlFlowGraph):
    result = [0]

    for block in cfg:
        result.append(len(block) + result[-1])

    return result


def _has_seen(seen: Dict[int, int], block: BasicBlock, current_stacksz: int) -> bool:
    has_seen = id(block) in seen
    if id(block) in seen:
        assert current_stacksz == seen[id(block)], \
            "Enter same block with different stack size"
    return has_seen


def _symbolic_eval(cfg: ControlFlowGraph) -> Dict[int, int]:
    result: Dict[int, int] = {}
    seen: Dict[int, Tuple[int, int]] = {}
    visit_stack = []
    offsets = _eval_offsets(cfg)
    current: BasicBlock = next(iter(cfg))
    stack_size = 0
    visit_stack.append((current, stack_size))

    result[-1] = 0
    while len(visit_stack) > 0:
        current, stack_size = visit_stack.pop()
        seen[id(current)] = stack_size
        instr_index_base = offsets[cfg.get_block_index(current)]

        for idx, instr in enumerate(current):
            if instr.has_jump():
                next_block = instr.arg
                assert isinstance(next_block, BasicBlock)
                stack_size_pro_jump = stack_size + instr.stack_effect(True)
                if not _has_seen(seen, next_block, stack_size_pro_jump):
                    visit_stack.append(
                        (next_block, stack_size_pro_jump))  # branch taken

            stack_size += instr.stack_effect(False)
            result[instr_index_base + idx] = stack_size

        next_block = current.next_block
        if next_block is not None and not _has_seen(seen, next_block, stack_size):
            visit_stack.append(
                (next_block, stack_size))  # fallthrough

    return result


def analyze_stack_top(
    func: FunctionType,
    last_instr: int,
):
    code = Bytecode.from_code(func.__code__)

    cfg = ControlFlowGraph.from_bytecode(code)

    eval_result = _symbolic_eval(cfg)[last_instr]

    return eval_result
