from pyckpt.interpreter.frame import NullObject as NullObject, fetch_exception as fetch_exception, restore_exception as restore_exception
from typing import Any, Generator

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
def resume_generator(generator, is_leaf, ret_val=..., exc_states=...) -> Any:
    """resume_generator(generator, is_leaf, ret_val=None, exc_states=None)
    resume_generator_pop(generator: Generator, is_leaf: bool, ret_val: Any) -> Tuple[Any, Optional[ExceptionStates]]

            Mimic CPython's behavior for generator evaluation
            See gen_send_ex2() in https://github.com/python/cpython/blob/3.11/Objects/genobject.c
    """
def snapshot_generator(generator: Generator) -> Any:
    """snapshot_generator(generator: Generator)"""
def snapshot_generator_frame(generator, analyzer) -> Any:
    """snapshot_generator_frame(generator, analyzer)"""
