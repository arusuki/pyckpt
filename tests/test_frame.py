# pylint: disable=C0104

import dis
import inspect
from typing import Optional

import dill
from pyckpt import analyzer
import pyckpt.frame

from pyckpt.frame import SavedFrame


def test_capture():
    BIG_NUMBER = 0x3f3f3f3f

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

    frame_: Optional[SavedFrame]

    def add(lhs, rhs):
        nonlocal frame_
        frame_ = SavedFrame.snapshot_from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top)
        return lhs + rhs

    assert add(1, 1) == 2
    assert frame_ is not None
    assert frame_.evaluate() == 2

    assert add(3, 4) == 7
    assert frame_ is not None
    assert frame_.evaluate() == 7


def test_frame_multiple_evaluation():

    def capture_frame():

        frame_ = SavedFrame.snapshot_from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top)

        return frame_, "hello"

    dis.dis(capture_frame)

    frame_, _ = capture_frame()

    ret_frame, ret = frame_.evaluate(frame_)

    assert ret == "hello", "invalid return value"
    assert ret_frame is frame_, "invalid resumed return value"

    try:
        ret_frame: SavedFrame
        ret_frame.evaluate(frame_)
        assert False, "allow multiple evaluation on one captured frame"
    except RuntimeError:
        pass


def test_save_frame():
    def capture_frame():

        frame_ = SavedFrame.snapshot_from_frame(
            inspect.currentframe(), False, analyzer.analyze_stack_top)

        return frame_, "hello"

    frame, ret1 = capture_frame()
    frame.registry = {}

    frame_s = dill.dumps(frame)

    new_frame: SavedFrame
    new_frame = dill.loads(frame_s)

    _, ret2 = new_frame.evaluate(frame)

    assert ret1 == ret2
