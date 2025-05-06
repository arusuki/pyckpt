from typing import Any, Generator

__test__: dict

def get_generator_type() -> type:
    """get_generator_type() -> Type"""
def is_suspended(generator: Generator) -> Any:
    """is_suspended(generator: Generator)"""
def make_generator(gen_states: dict, frame_states: dict) -> Any:
    """make_generator(gen_states: Dict, frame_states: Dict)"""
def snapshot_generator(generator: Generator) -> Any:
    """snapshot_generator(generator: Generator)"""
def snapshot_generator_frame(generator, analyzer) -> Any:
    """snapshot_generator_frame(generator, analyzer)"""
