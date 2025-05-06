import inspect
from types import FrameType
from typing import Generator

import dill
import pytest

from pyckpt.analyzer import analyze_stack_top
from pyckpt.interpreter import frame as _frame
from pyckpt.interpreter import generator as _generator


def test_snapshot_generator():
    def foo():
        frame = inspect.currentframe()
        yield frame

    gen = foo()
    assert isinstance(gen, Generator)
    captured = _generator.snapshot_generator(gen)

    assert captured["gi_code"] is foo.__code__
    assert "foo" in captured["gi_name"]
    assert "foo" in captured["gi_qualname"]


def test_make_generator():
    def foo():
        frame = inspect.currentframe()
        yield frame
        yield 42

    gen = foo()
    frame = next(gen)
    gen_ret = _frame.get_generator(frame)
    assert gen_ret is gen

    gen_states = _generator.snapshot_generator(gen_ret)
    frame_states = _frame.snapshot(frame, False, analyze_stack_top)
    for i, obj in enumerate(frame_states["stack"]):
        if isinstance(obj, FrameType):
            frame_states["stack"][i] = None
    for i, obj in enumerate(frame_states["nlocals"]):
        if isinstance(obj, FrameType):
            frame_states["nlocals"][i] = None
    del frame_states["generator"]
    frame_states = dill.copy(frame_states)
    gen_new = _generator.make_generator(gen_states, frame_states)
    assert isinstance(gen_new, Generator)
    assert _generator.is_suspended(gen_new)
    gen_states = _generator.snapshot_generator(gen_ret)
    with pytest.raises(StopIteration):
        assert next(gen_new) == 42
        next(gen_new)
    with pytest.raises(StopIteration):
        assert next(gen) == 42
        next(gen)


def test_get_generator_type():
    def test():
        yield 42

    gen = test()
    gen_type = _generator.get_generator_type()
    assert isinstance(gen, gen_type)
    assert type(gen) is gen_type


def test_snapshot_generator_frame():
    def foo():
        yield 41
        yield 42

    gen = foo()
    assert next(gen) == 41

    gen_states = _generator.snapshot_generator(gen)
    assert gen_states["suspended"]
    frame_states = _generator.snapshot_generator_frame(gen, analyze_stack_top)
    new_gen = _generator.make_generator(gen_states, frame_states)
    with pytest.raises(StopIteration):
        assert next(gen) == 42
        next(gen)
    with pytest.raises(StopIteration):
        assert next(new_gen) == 42
        next(new_gen)
