from builtins import FrameType, FunctionType
from threading import Thread
from typing import Analyzer, Any, ExceptionStates

CALL_CODES: list
CALL_INSTR_NAMES: list
NullObject: NullObjectType
__pyx_capi__: dict
__test__: dict

class NullObjectType: ...

def eval_frame_at_lasti(func_obj: FunctionType, nlocals: list[Any], stack: list[Any], is_leaf: bool, ret_value: Any = ..., prev_instr_offset=..., exc_states: ExceptionStates | None = ...) -> tuple[Any, ExceptionStates | None]:
    """eval_frame_at_lasti(func_obj: FunctionType, nlocals: List[Any], stack: List[Any], is_leaf: bool, ret_value: Any = None, prev_instr_offset=-1, exc_states: Optional[ExceptionStates] = None) -> Tuple[Any, Optional[ExceptionStates]]"""
def frame_specials_size() -> Any:
    """frame_specials_size()"""
def get_generator(frame: FrameType) -> Any:
    """get_generator(frame: FrameType)"""
def is_call_instr(opcode: int) -> Any:
    """is_call_instr(opcode: int)"""
def restore_exception(states: ExceptionStates) -> Any:
    """restore_exception(states: ExceptionStates)"""
def restore_thread_state(state: dict) -> Any:
    """restore_thread_state(state: Dict)"""
def save_thread_state(thread: Thread) -> Any:
    """save_thread_state(thread: Thread)"""
def snapshot(frame_obj: FrameType, is_leaf: bool, analyzer: Analyzer) -> dict:
    """snapshot(frame_obj: FrameType, is_leaf: bool, analyzer: Analyzer) -> Dict"""
