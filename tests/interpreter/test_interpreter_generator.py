import inspect
from types import FrameType, FunctionType
from typing import Generator

import dill
import pytest

from pyckpt.analyzer import analyze_stack_top
from pyckpt.interpreter import frame as _frame
from pyckpt.interpreter import generator as _generator


def _make_new_generator_from_function(func: FunctionType):
    return _generator.make_new_generator(
        func_code=func.__code__,
        func_name=func.__name__,
        func_qualname=func.__qualname__,
    )


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


def test_snapshot_generator_with_exception():
    def foo():
        try:
            raise RuntimeError("test")
        except RuntimeError:
            yield 41
            yield 42

    gen = foo()
    assert isinstance(gen, Generator)
    assert next(gen) == 41
    captured = _generator.snapshot_generator(gen)
    assert isinstance(captured["exception"], RuntimeError)


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


def test_make_new_generator():
    def not_generator():
        return 42

    def foo():
        yield 42

    with pytest.raises(ValueError):
        _make_new_generator_from_function(not_generator)

    not_code = None
    with pytest.raises(ValueError):
        _generator.make_new_generator(not_code, "test", "test_make_new_generator.test")

    new_gen = _make_new_generator_from_function(foo)

    with pytest.raises(StopIteration):
        next(new_gen)


def test_setup_generator():
    def foo():
        yield 41
        yield 42

    def bar():
        yield 41
        yield 42

    gen = foo()
    assert next(gen) == 41
    gen_states = _generator.snapshot_generator(gen)
    frame_states = _generator.snapshot_generator_frame(gen, analyze_stack_top)
    new_gen = _make_new_generator_from_function(foo)
    _generator.setup_generator(new_gen, gen_states, frame_states)

    assert next(new_gen) == 42
    with pytest.raises(StopIteration):
        next(new_gen)

    assert next(gen) == 42
    with pytest.raises(StopIteration):
        next(gen)

    wrong_gen = bar()
    assert next(wrong_gen) == 41
    gen_states = _generator.snapshot_generator(wrong_gen)
    frame_states = _generator.snapshot_generator_frame(wrong_gen, analyze_stack_top)

    new_gen = _make_new_generator_from_function(foo)
    with pytest.raises(ValueError):
        _generator.setup_generator(new_gen, gen_states, frame_states)


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


def test_resume_generator():
    def foo():
        yield 41
        yield 42

    gen = foo()
    assert next(gen) == 41

    ret, exc = _generator.resume_generator(gen, False, None)
    assert exc is None
    assert ret == 42


def test_generator_execution_with_exception():
    def foo():
        try:
            raise RuntimeError("test")
        except RuntimeError:
            yield 41
            yield 42
            raise

    gen = foo()
    assert next(gen) == 41

    gen_states = _generator.snapshot_generator(gen)
    assert isinstance(gen_states["exception"], RuntimeError)
    frame_states = _generator.snapshot_generator_frame(gen, analyze_stack_top)
    assert gen_states["gi_frame_state"] == -1
    # continue execution
    new_gen = _generator.make_generator(gen_states, frame_states)
    ret, exc = _generator.resume_generator(new_gen, False, None)
    assert ret == 42
    assert exc is None
    # reraise exception
    with pytest.raises(RuntimeError):
        next(new_gen)
    with pytest.raises(StopIteration):
        next(new_gen)


def test_resume_return_value():
    def foo():
        x = yield 42
        assert x == "42"
        # return None <-- implicit return

    # first starts the generator
    gen = foo()
    assert next(gen) == 42
    # then resume
    ret, exc = _generator.resume_generator(gen, False, "43")
    assert ret is _frame.NullObject
    assert exc is not None
    assert isinstance(exc[1], AssertionError)
    del gen

    gen = foo()
    assert next(gen) == 42
    ret, exc = _generator.resume_generator(gen, False, "42")
    assert ret is None
    assert exc is not None
    assert isinstance(exc[1], StopIteration)


def test_resume_with_exception():
    def foo():
        yield 42
        # return None <-- implicit return

    gen = foo()
    assert next(gen) == 42
    ret, exc = _generator.resume_generator(
        gen,
        False,
        None,
        (RuntimeError, RuntimeError("42"), None),
    )
    assert ret is _frame.NullObject
    assert exc is not None
    assert isinstance(exc[1], RuntimeError)


def test_reduce_null_object_type():
    obj = {"null": _frame.NullObject}

    with pytest.raises(NotImplementedError):
        dill.copy(obj)
