# pylint: disable=C0104
import inspect
import threading
from ast import FunctionType
from typing import Optional

import dill
import pytest

import pyckpt.frame
from pyckpt import analyzer, interpreter
from pyckpt.frame import FrameCocoon, LiveFunctionFrame, LiveGeneratorFrame
from pyckpt.objects import SnapshotContextManager, SpawnContextManager

BIG_NUMBER = 0x3F3F3F3F


def test_capture():
    def foo():
        _x = 1
        _y = 2
        frame = pyckpt.frame.capture(0)
        assert "foo" in frame.f_code.co_name

        frame = pyckpt.frame.capture(1)
        assert "bar" in frame.f_code.co_name

        frame = pyckpt.frame.capture(2)
        assert "test_capture" in frame.f_code.co_name

        try:
            frame = pyckpt.frame.capture(BIG_NUMBER)
            assert False, "allow invalid backtrace number in pyckpt.frame.capture()"
        except ValueError:
            pass

    def bar():
        _z = 3
        foo()

    bar()


def test_capture_with_analyzer():
    frame_: Optional[FrameCocoon] = None

    def add(lhs, rhs):
        nonlocal frame_
        ctxs = SnapshotContextManager()
        frame_ = FrameCocoon.snapshot_from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top, ctxs
        )
        return lhs + rhs

    assert add(1, 1) == 2
    assert isinstance(frame_, FrameCocoon)
    cocoon = frame_.clone()
    result, err = cocoon.spawn({}).evaluate()
    assert result == 2
    assert err is None

    # try different arguments
    assert add(3, 4) == 7
    cocoon = frame_.clone()
    assert isinstance(frame_, FrameCocoon)
    result, err = cocoon.spawn({}).evaluate()
    assert result == 7
    assert err is None


def test_frame_multiple_evaluation():
    def capture_frame():
        ctxs = SnapshotContextManager()
        cocoon = FrameCocoon.snapshot_from_frame(
            inspect.currentframe(),
            False,
            analyzer.analyze_stack_top,
            ctxs,
        )
        return cocoon, "hello"

    cocoon, _ = capture_frame()
    frame_ = cocoon.spawn({})
    (ret_frame, ret), exc_states = frame_.evaluate(frame_)

    assert exc_states is None
    assert ret_frame is frame_
    assert ret == "hello"

    with pytest.raises(ValueError):
        ret_frame.evaluate(frame_)


def test_save_cocoon():
    def capture_frame():
        ctxs = SnapshotContextManager()
        cocoon = FrameCocoon.snapshot_from_frame(
            inspect.currentframe(),
            False,
            analyzer.analyze_stack_top,
            ctxs,
        )
        return cocoon, "hello"

    c1, ret1 = capture_frame()
    c1_s = dill.dumps(c1)
    c2: FrameCocoon
    c2 = dill.loads(c1_s)
    (c3, ret2), err = c2.spawn({}).evaluate(c1)

    assert err is None
    assert c3 is c1
    assert ret1 == ret2


def test_seg():
    def capture_frame():
        ctxs = SnapshotContextManager()
        cocoon = FrameCocoon.snapshot_from_frame(
            inspect.currentframe(),
            False,
            analyzer.analyze_stack_top,
            ctxs,
        )
        return cocoon, "hello"

    c1, ret1 = capture_frame()
    c1_s = dill.dumps(c1)

    c2: FrameCocoon
    c2 = dill.loads(c1_s)
    (c3, ret2), err = c2.spawn({}).evaluate(c1)

    assert c3 is c1
    assert ret1 == ret2
    assert err is None


def test_raise_exception():
    outer_cocoon: Optional[FrameCocoon] = None
    inner_cocoon: Optional[FrameCocoon] = None
    ctxs = SnapshotContextManager()

    def outer():
        nonlocal outer_cocoon
        try:
            outer_cocoon = FrameCocoon.snapshot_from_frame(
                inspect.currentframe(),
                False,
                analyzer.analyze_stack_top,
                ctxs,
            )
            inner()
        except RuntimeError as e:
            assert str(e) == "test"
        return True

    def inner():
        nonlocal inner_cocoon
        inner_cocoon = FrameCocoon.snapshot_from_frame(
            inspect.currentframe(),
            False,
            analyzer.analyze_stack_top,
            ctxs,
        )
        raise RuntimeError("test")

    def evaluate(new_frames):
        inner_ret, err = new_frames[1].spawn({}).evaluate()
        assert inner_ret is None
        assert isinstance(err, tuple)

        outer_ret, err = new_frames[0].spawn({}).evaluate(inner_ret)
        assert isinstance(outer_ret, bool) and outer_ret
        assert err is None

    assert outer()
    assert outer_cocoon is not None
    assert inner_cocoon is not None

    frames = [outer_cocoon, inner_cocoon]
    evaluate(dill.copy(frames))  # first
    evaluate(dill.copy(frames))  # second


def test_handled_exception():
    ctxs = SnapshotContextManager()

    def capture_frame():
        try:
            raise RuntimeError("JB")
        except RuntimeError:
            cocoon = FrameCocoon.snapshot_from_frame(
                inspect.currentframe(),
                False,
                analyzer.analyze_stack_top,
                ctxs,
            )
            ts = interpreter.save_thread_state(threading.current_thread())
            return (cocoon, ts), "hello"

    (c1, ts), ret1 = capture_frame()
    c2 = c1.clone()
    interpreter.restore_thread_state(ts)
    c2.spawn({}).evaluate(c2)

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
    ctxs = SnapshotContextManager()

    def generator_function():
        c = FrameCocoon.snapshot_from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top, ctxs
        )
        if c:
            yield c
        else:
            yield 1
            yield 2

    # Create a snapshot of the generator frame
    gen = generator_function()
    cocoon = next(gen).clone()
    assert isinstance(cocoon, FrameCocoon)
    with pytest.raises(StopIteration):
        next(gen)
    # Spawn a LiveGeneratorFrame from the cocoon
    spawn_ctxs = SpawnContextManager()
    spawn_ctxs.register_object(id(gen), _make_new_generator_from_function(cocoon.func))
    live_frame = cocoon.spawn(spawn_ctxs)
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


def test_spawn_with_function_frame():
    def test_function(a, b):
        return a + b

    # Create a snapshot of the function frame
    cocoon = FrameCocoon(
        is_leaf=True,
        func=test_function,
        stack=[],
        nlocals=[3, 4],
        prev_instr_offset=-1,
        generator=None,
    )

    # Spawn a LiveFunctionFrame from the cocoon
    contexts = SpawnContextManager()
    live_frame = cocoon.spawn(contexts)
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
    ctxs = SnapshotContextManager()

    def generator_function():
        frame = inspect.currentframe()
        yield frame
        yield 42

    gen = generator_function()

    # Advance the generator to capture its frame
    frame = next(gen)

    # Create a snapshot from the generator frame
    cocoon = FrameCocoon.snapshot_from_frame(
        frame=frame,
        is_leaf=False,
        stack_analyzer=analyzer.analyze_stack_top,
        contexts=ctxs,
    )

    assert isinstance(cocoon, FrameCocoon)
    assert cocoon.is_leaf is False
    assert cocoon.func == generator_function
    assert cocoon.generator is not None

    # Spawn a LiveGeneratorFrame from the cocoon
    spawn_ctxs = SpawnContextManager()
    spawn_ctxs.register_object(id(gen), _make_new_generator_from_function(cocoon.func))
    live_frame = cocoon.spawn(spawn_ctxs)
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
