from pyckpt.interpreter.frame import NullObject as NullObject
from typing import Any, Generator

__test__: dict

class ClearFrame(Exception): ...

def get_generator_type() -> type:
    """get_generator_type() -> Type"""
def is_suspended(generator: Generator) -> Any:
    """is_suspended(generator: Generator)"""
def make_generator(gen_states: dict, frame_states: dict) -> Any:
    """make_generator(gen_states: Dict, frame_states: Dict)"""
def pop_generator_exception(generator) -> Any:
    """pop_generator_exception(generator)"""
def push_generator_exception(generator) -> Any:
    """push_generator_exception(generator)"""
def resume_generator_pop(generator, is_leaf, ret_val=...) -> Any:
    """resume_generator_pop(generator, is_leaf, ret_val=None)
    resume_generator_pop(generator: Generator, is_leaf: bool, ret_val: Any) -> Tuple[Any, Optional[ExceptionStates]]

            Mimic CPython's behavior for generator evaluation
            See gen_send_ex2() in https://github.com/python/cpython/blob/3.11/Objects/genobject.c
    """
def snapshot_generator(generator: Generator) -> Any:
    """snapshot_generator(generator: Generator)"""
def snapshot_generator_frame(generator, analyzer) -> Any:
    """snapshot_generator_frame(generator, analyzer)"""
