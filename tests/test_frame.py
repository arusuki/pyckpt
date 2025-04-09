# pylint: disable=C0104
import inspect
import threading
from typing import Optional

import dill

import pyckpt.frame
from pyckpt import analyzer, interpreter
from pyckpt.frame import FrameCocoon

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
        frame_ = FrameCocoon.from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top, {}, {}
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
        cocoon = FrameCocoon.from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top, {}, {}
        )
        return cocoon, "hello"

    cocoon, _ = capture_frame()
    frame_ = cocoon.spawn({})
    (ret_frame, ret), exc_states = frame_.evaluate(frame_)

    assert exc_states is None
    assert ret_frame is frame_
    assert ret == "hello"

    try:
        ret_frame.evaluate(frame_)
        assert False, "allow multiple evaluation on one captured frame"
    except RuntimeError:
        pass


def test_save_cocoon():
    def capture_frame():
        cocoon = FrameCocoon.from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top, {}, {}
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
        cocoon = FrameCocoon.from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top, {}, {}
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

    def outer():
        nonlocal outer_cocoon
        try:
            outer_cocoon = FrameCocoon.from_frame(
                inspect.currentframe(), False, analyzer.analyze_stack_top, {}, {}
            )
            inner()
        except RuntimeError as e:
            assert str(e) == "test"
        return True

    def inner():
        nonlocal inner_cocoon
        inner_cocoon = FrameCocoon.from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top, {}, {}
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
    def capture_frame():
        try:
            raise RuntimeError("JB")
        except RuntimeError:
            cocoon = FrameCocoon.from_frame(
                inspect.currentframe(), False, analyzer.analyze_stack_top, {}, {}
            )
            ts = interpreter.save_thread_state(threading.current_thread())
            return (cocoon, ts), "hello"

    (c1, ts), ret1 = capture_frame()
    c2 = c1.clone()
    interpreter.restore_thread_state(ts)
    c2.spawn({}).evaluate(c2)

    assert ret1 == "hello"
