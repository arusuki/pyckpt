from pyckpt.interpreter.frame import NullObject as NullObject, fetch_exception as fetch_exception, restore_exception as restore_exception
from typing import Any, Generator, ExceptionStates, Optional

__test__: dict

class ClearFrame(Exception): ...

def get_generator_type() -> type:
    """get_generator_type() -> Type"""
def is_executing(generator: Generator) -> Any:
    """is_executing(generator: Generator)"""
def is_suspended(generator: Generator) -> Any:
    """is_suspended(generator: Generator)"""
def make_generator(gen_states: dict, frame_states: dict) -> Any:
    """make_generator(gen_states: Dict, frame_states: Dict)"""
def make_new_generator(func_code, func_name, func_qualname) -> Any:
    """make_new_generator(func_code, func_name, func_qualname)"""
def resume_generator(generator, is_leaf, ret_val=..., exc_states=...) -> Any:
    """resume_generator(generator, is_leaf, ret_val=None, exc_states=None)
    resume_generator_pop(generator: Generator, is_leaf: bool, ret_val: Any) -> Tuple[Any, Optional[ExceptionStates]]

            Mimic CPython's behavior for generator evaluation
            See gen_send_ex2() in https://github.com/python/cpython/blob/3.11/Objects/genobject.c
    """
def setup_generator(generator: Generator, gen_states: dict, frame_states: dict) -> Any:
    """setup_generator(generator: Generator, gen_states: Dict, frame_states: Dict)"""
def snapshot_generator(generator: Generator) -> Any:
    """snapshot_generator(generator: Generator)"""

def snapshot_frame_generator(generator: Generator, stack_size: int = -1): ...

def generator_push_stack(generator: Generator, value: Any): ...

def generator_shrink_stack(generator: Generator, stack_size: int): ...

def generator_set_instr_offset(generator: Generator, instr_offset: int): ...

def generator_resume(generator: Generator, exc_states: Optional[ExceptionStates] = None): ...
