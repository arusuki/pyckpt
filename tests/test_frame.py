import inspect
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Generator, Optional

import pytest

from pyckpt import interpreter, objects
from pyckpt.frame import FunctionFrame, GeneratorFrame, snapshot_frame
from pyckpt.interpreter.frame import NullObject


def test_snapshot_frame():
    frame_: Optional[FunctionFrame] = None

    def add(lhs, rhs):
        nonlocal frame_
        frame_ = snapshot_frame(inspect.currentframe())
        return lhs + rhs

    assert add(1, 1) == 2
    frame_, _ = objects.copy(frame_)
    assert isinstance(frame_, FunctionFrame)
    assert frame_._requires_ret

    result, err = frame_.return_value(None).evaluate()
    assert result == 2
    assert err is None

    # try different arguments
    assert add(3, 4) == 7
    frame_, _ = objects.copy(frame_)
    assert isinstance(frame_, FunctionFrame)
    result, err = frame_.return_value(None).evaluate()
    assert result == 7
    assert err is None


def test_function_frame_evaluation():
    def capture_frame():
        frame_ = snapshot_frame(inspect.currentframe())
        return frame_, "hello"

    frame_, _ = capture_frame()
    with pytest.raises(ValueError):
        frame_.evaluate()
    (ret_frame, ret), exc_states = frame_.return_value(frame_).evaluate()
    assert exc_states is None
    assert ret_frame is frame_
    assert ret == "hello"

    with pytest.raises(ValueError):
        ret_frame.return_value(frame_).evaluate()


def test_function_frame_copy():
    def capture_frame():
        cocoon = snapshot_frame(inspect.currentframe())
        return cocoon, "hello"

    frame_, ret1 = capture_frame()
    frame_1, _ = objects.copy(frame_)
    (frame_2, ret2), err = frame_1.return_value(frame_).evaluate()

    assert err is None
    assert frame_2 is frame_
    assert ret1 == ret2

def test_function_frame_eval_exception():
    outer_frame: Optional[FunctionFrame] = None
    inner_frame: Optional[FunctionFrame] = None

    def outer():
        nonlocal outer_frame
        try:
            outer_frame = snapshot_frame(inspect.currentframe())
            inner()
        except RuntimeError as e:
            assert str(e) == "test"
        return True

    def inner():
        nonlocal inner_frame
        inner_frame = snapshot_frame(inspect.currentframe())
        raise RuntimeError("test")

    def evaluate(new_frames):
        inner_ret, err = new_frames[1].return_value(None).evaluate()
        assert inner_ret is NullObject
        assert isinstance(err, tuple)

        outer_ret, err = new_frames[0].return_value(inner_ret).evaluate()
        assert isinstance(outer_ret, bool) and outer_ret
        assert err is None

    assert outer()
    assert isinstance(outer_frame, FunctionFrame)
    assert isinstance(inner_frame, FunctionFrame)

    frames = [outer_frame, inner_frame]
    evaluate(objects.copy(frames)[0])  # first
    evaluate(objects.copy(frames)[0])  # second

def test_function_frame_eval_with_thread_state():
    def capture_frame():
        try:
            raise RuntimeError("test exception")
        except RuntimeError:
            frame_ = snapshot_frame(inspect.currentframe())
            ts = interpreter.save_thread_state(threading.current_thread())
            return (frame_, ts), "hello"

    (frame_, ts), ret1 = capture_frame()
    frame_1, _ = objects.copy(frame_)
    interpreter.restore_thread_state(ts)
    frame_1.return_value(None).evaluate()
    assert ret1 == "hello"


def generator_frame_snapshot(gi_frame: FrameType):
    def copy_generator(gen: Generator) -> Generator:
        cons, args = objects.reduce_generator_v1(gen)
        return cons(*args)

    f = snapshot_frame(gi_frame)
    assert isinstance(f, GeneratorFrame)
    f.generator = copy_generator(f.generator)
    return f


def test_generator_frame_eval():
    def generator_function():
        f = generator_frame_snapshot(inspect.currentframe())
        yield f
        yield 1
        yield 2

    # Create a snapshot of the generator frame
    gen = generator_function()
    frame_ = next(gen)
    # Spawn a LiveGeneratorFrame from the frame_
    assert isinstance(frame_, GeneratorFrame)

    # Evaluate the generator frame
    gen = frame_.generator
    result, exc_states = frame_.return_value(None).evaluate()
    assert exc_states is None
    assert result is None

    assert tuple(next(gen) for _ in range(2)) ==  (1, 2)

def test_generator_frame_stop_iteration():
    def generator_function():
        f = generator_frame_snapshot(inspect.currentframe())
        if f:
            yield f
        return "return"

    # Create a snapshot of the generator frame
    gen = generator_function()
    frame_ = next(gen)
    # Spawn a LiveGeneratorFrame from the frame_
    assert isinstance(frame_, GeneratorFrame)

    # Evaluate the generator frame
    gen = frame_.generator
    result, exc_states = frame_.return_value(None).evaluate()

    assert result == "return"
    assert exc_states is not None
    assert isinstance(exc_states[1], StopIteration)


@contextmanager
def my_context_mgr():
    yield "manager"
    print("finished")


def test_snapshot_function_frame_with_context_manager(capsys):
    def foo():
        with my_context_mgr() as s:
            frame_ = snapshot_frame(inspect.currentframe())
            if frame_:
                new_frame_, _ = objects.copy(frame_)
                assert new_frame_.states.stack[0] is not frame_.states.stack[0]
                return new_frame_
            else:
                return s

    f = foo()
    assert isinstance(f, FunctionFrame)
    result = capsys.readouterr()
    assert result.out.count("finished") == 1

    ret, exc = f.return_value(None).evaluate()
    assert ret == "manager"
    assert exc is None
    result = capsys.readouterr()
    assert result.out.count("finished") == 1
