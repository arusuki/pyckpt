# pylint: disable=C0104
import inspect
import threading
from ast import FunctionType
from contextlib import contextmanager
from typing import Generator, Optional

import pytest

import pyckpt.frame
from pyckpt import analyzer, interpreter, objects
from pyckpt.frame import (
    FunctionFrameCocoon,
    GeneratorFrameCocoon,
    LiveFunctionFrame,
    LiveGeneratorFrame,
    snapshot_from_frame,
)
from pyckpt.interpreter.frame import NullObject

BIG_NUMBER = 0x3F3F3F3F


def test_capture_with_analyzer():
    frame_: Optional[FunctionFrameCocoon] = None

    def add(lhs, rhs):
        nonlocal frame_
        frame_ = snapshot_from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top
        )
        return lhs + rhs

    assert add(1, 1) == 2
    assert isinstance(frame_, FunctionFrameCocoon)
    cocoon, _ = objects.copy(frame_)
    result, err = cocoon.spawn().evaluate()
    assert result == 2
    assert err is None

    # try different arguments
    assert add(3, 4) == 7
    cocoon, _ = objects.copy(frame_)
    assert isinstance(frame_, FunctionFrameCocoon)
    result, err = cocoon.spawn().evaluate()
    assert result == 7
    assert err is None


def test_frame_multiple_evaluation():
    def capture_frame():
        cocoon = snapshot_from_frame(
            inspect.currentframe(),
            False,
            analyzer.analyze_stack_top,
        )
        return cocoon, "hello"

    cocoon, _ = capture_frame()
    frame_ = cocoon.spawn()
    (ret_frame, ret), exc_states = frame_.evaluate(frame_)

    assert exc_states is None
    assert ret_frame is frame_
    assert ret == "hello"

    with pytest.raises(ValueError):
        ret_frame.evaluate(frame_)


def test_save_cocoon():
    def capture_frame():
        cocoon = snapshot_from_frame(
            inspect.currentframe(),
            False,
            analyzer.analyze_stack_top,
        )
        return cocoon, "hello"

    c1, ret1 = capture_frame()
    c2, _ = objects.copy(c1)
    (c3, ret2), err = c2.spawn().evaluate(c1)

    assert err is None
    assert c3 is c1
    assert ret1 == ret2


def test_seg():
    def capture_frame():
        cocoon = snapshot_from_frame(
            inspect.currentframe(),
            False,
            analyzer.analyze_stack_top,
        )
        return cocoon, "hello"

    c1, ret1 = capture_frame()
    c2, _ = objects.copy(c1)
    assert isinstance(c2, FunctionFrameCocoon)
    (c3, ret2), err = c2.spawn().evaluate(c1)

    assert c3 is c1
    assert ret1 == ret2
    assert err is None


def test_raise_exception():
    outer_cocoon: Optional[FunctionFrameCocoon] = None
    inner_cocoon: Optional[FunctionFrameCocoon] = None

    def outer():
        nonlocal outer_cocoon
        try:
            outer_cocoon = snapshot_from_frame(
                inspect.currentframe(),
                False,
                analyzer.analyze_stack_top,
            )
            inner()
        except RuntimeError as e:
            assert str(e) == "test"
        return True

    def inner():
        nonlocal inner_cocoon
        inner_cocoon = snapshot_from_frame(
            inspect.currentframe(),
            False,
            analyzer.analyze_stack_top,
        )
        raise RuntimeError("test")

    def evaluate(new_frames):
        inner_ret, err = new_frames[1].spawn().evaluate()
        assert inner_ret is NullObject
        assert isinstance(err, tuple)

        outer_ret, err = new_frames[0].spawn().evaluate(inner_ret)
        assert isinstance(outer_ret, bool) and outer_ret
        assert err is None

    assert outer()
    assert outer_cocoon is not None
    assert inner_cocoon is not None

    frames = [outer_cocoon, inner_cocoon]
    evaluate(objects.copy(frames)[0])  # first
    evaluate(objects.copy(frames)[0])  # second


def test_handled_exception():
    def capture_frame():
        try:
            raise RuntimeError("JB")
        except RuntimeError:
            cocoon = snapshot_from_frame(
                inspect.currentframe(),
                False,
                analyzer.analyze_stack_top,
            )
            ts = interpreter.save_thread_state(threading.current_thread())
            return (cocoon, ts), "hello"

    (c1, ts), ret1 = capture_frame()
    c2, _ = objects.copy(c1)
    interpreter.restore_thread_state(ts)
    c2.spawn().evaluate(c2)

    assert ret1 == "hello"


def test_live_generator_frame_evaluation():
    def generator_function():
        yield 1
        yield 2
        return 3

    gen = generator_function()
    live_gen_frame = pyckpt.frame.LiveGeneratorFrame(gen, is_leaf=False)

    # starts the generator
    assert next(gen) == 1

    result, exc_states = live_gen_frame._evaluate(None, None)
    assert result == 2
    assert exc_states is None

    result, exc_states = live_gen_frame._evaluate(None, None)
    assert result == 3
    assert exc_states is not None
    assert isinstance(exc_states[1], StopIteration)

    # Test cleanup
    live_gen_frame._cleanup()
    with pytest.raises(AttributeError):
        _ = live_gen_frame._gen
    with pytest.raises(AttributeError):
        _ = live_gen_frame._is_leaf


def _make_new_generator_from_function(func: FunctionType):
    return interpreter.make_new_generator(
        func_code=func.__code__,
        func_name=func.__name__,
        func_qualname=func.__qualname__,
    )


def test_spawn_with_generator_frame():
    def generator_function():
        c = snapshot_from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top
        )
        yield c
        yield 1
        yield 2

    # Create a snapshot of the generator frame
    gen = generator_function()
    cocoon = next(gen)
    cocoon, _ = objects.copy(cocoon)
    # Spawn a LiveGeneratorFrame from the cocoon
    live_frame = cocoon.spawn()
    assert isinstance(live_frame, LiveGeneratorFrame)

    # Evaluate the generator frame
    result, exc_states = live_frame._evaluate(None, None)
    assert result == 1
    assert exc_states is None

    result, exc_states = live_frame._evaluate(None, None)
    assert result == 2
    assert exc_states is None

    result, exc_states = live_frame.evaluate()
    assert result is None
    assert exc_states is not None
    assert isinstance(exc_states[1], StopIteration)

    # Test cleanup
    with pytest.raises(AttributeError):
        _ = live_frame._gen
    with pytest.raises(AttributeError):
        _ = live_frame._is_leaf


@contextmanager
def my_context_mgr():
    yield "manager"
    print("finished")


def test_snapshot_with_context_manager(capsys):
    def foo():
        with my_context_mgr() as s:
            frame_ = snapshot_from_frame(
                inspect.currentframe(), False, analyzer.analyze_stack_top
            )
            if frame_:
                new_frame_, _ = objects.copy(frame_)
                assert new_frame_.stack[0] is not frame_.stack[0]
                return new_frame_
            else:
                return s

    f = foo()
    assert isinstance(f, FunctionFrameCocoon)
    result = capsys.readouterr()
    assert result.out.count("finished") == 1
    print(f.stack)

    ret, exc = f.spawn().evaluate()
    assert ret == "manager"
    assert exc is None
    result = capsys.readouterr()
    assert result.out.count("finished") == 1


def test_spawn_with_function_frame():
    def test_function(a, b):
        return a + b

    # Create a snapshot of the function frame
    cocoon = FunctionFrameCocoon(
        is_leaf=True,
        is_return=False,
        func=test_function,
        stack=[],
        nlocals=[3, 4],
        prev_instr_offset=-1,
    )

    # Spawn a LiveFunctionFrame from the cocoon
    live_frame = cocoon.spawn()
    assert isinstance(live_frame, LiveFunctionFrame)

    # Evaluate the function frame
    result, exc_states = live_frame.evaluate()
    assert result == 7
    assert exc_states is None

    # Test cleanup
    with pytest.raises(AttributeError):
        _ = live_frame.func
    with pytest.raises(AttributeError):
        _ = live_frame.stack


def test_snapshot_from_generator_frame():
    def generator_function():
        frame = inspect.currentframe()
        yield frame
        yield 42

    gen = generator_function()

    # Advance the generator to capture its frame
    frame = next(gen)

    # Create a snapshot from the generator frame
    cocoon = snapshot_from_frame(
        frame=frame,
        is_leaf=False,
        stack_analyzer=analyzer.analyze_stack_top,
    )

    assert isinstance(cocoon, GeneratorFrameCocoon)
    assert cocoon.is_leaf is False
    assert isinstance(cocoon.gen, Generator)

    # Spawn a LiveGeneratorFrame from the cocoon
    live_frame = cocoon.spawn()
    assert isinstance(live_frame, LiveGeneratorFrame)

    # Evaluate the generator frame
    result, exc_states = live_frame._evaluate(None, None)
    assert result == 42
    assert exc_states is None

    # Test cleanup
    live_frame._cleanup()
    with pytest.raises(AttributeError):
        _ = live_frame._gen
    with pytest.raises(AttributeError):
        _ = live_frame._is_leaf
