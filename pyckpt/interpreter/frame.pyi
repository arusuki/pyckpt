from builtins import FrameType, FunctionType
from threading import Thread
from types import CodeType, TracebackType
from typing import Analyzer, Any, NamedTuple, Optional, Type

CALL_CODES: list
CALL_INSTR_NAMES: list
NullObject: NullObjectType
__pyx_capi__: dict
__test__: dict

class NullObjectType:
    def __reduce__(self) -> str | tuple[Any, ...]:
        """__reduce__(self) -> str | tuple[Any, ...]"""

ExceptionStates = tuple[Type, Exception, TracebackType]

class EvaluateResult(NamedTuple):
    ret: Any
    exception_states: Optional[ExceptionStates]

def fetch_exception() -> ExceptionStates:
    """fetch_exception() -> ExceptionStates"""

def frame_specials_size() -> Any:
    """frame_specials_size()"""

def get_generator(frame: FrameType) -> Any:
    """get_generator(frame: FrameType)"""

def restore_exception(states: ExceptionStates) -> Any:
    """restore_exception(states: ExceptionStates)"""

def restore_thread_state(state: dict) -> Any:
    """restore_thread_state(state: Dict)"""

def save_thread_state(thread: Thread) -> Any:
    """save_thread_state(thread: Thread)"""

def set_profile(func: FunctionType | None) -> Any: ...
def set_profile_all_threads(func: FunctionType | None) -> Any:
    """set_profile_all_threads(func: Optional[FunctionType])"""

def snapshot_frame(frame: FrameType, stack_size_hint: int = -1) -> dict: ...
def eval_frame(
    func: FunctionType,
    nlocals: list[Any],
    stack: list[Any],
    instr_offset: int,
    exc_states: ExceptionStates | None = None,
) -> tuple[Any, ExceptionStates | None]: ...
def frame_lasti_opcode(frame: FrameType) -> int: ...
