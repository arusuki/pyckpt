import inspect
import unittest
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from pyckpt.analyzer import analyze_stack_top
from pyckpt.frame import FrameCocoon, LiveFunctionFrame
from pyckpt.generator import (
    GeneratorContext,
    GeneratorFrameStatesCocoon,
)
from pyckpt.objects import CRContextCocoon, SnapshotContextManager, SpawnContextManager


class TestGeneratorFrameStatesCocoon(unittest.TestCase):
    def setUp(self):
        self.states = {
            "func": lambda x: x,
            "nlocals": [1, 2, 3],
            "stack": [4, 5, 6],
            "prev_instr_offset": 10,
            "is_leaf": True,
        }

    @patch("pyckpt.generator.snapshot_objects")
    def test_snapshot_from_states(self, mock_snapshot_objects):
        mock_snapshot_objects.side_effect = lambda x, _: x
        snapshot_ctxs = SnapshotContextManager()
        cocoon = GeneratorFrameStatesCocoon.snapshot_from_states(
            self.states, snapshot_ctxs
        )
        self.assertEqual(cocoon.func, self.states["func"])
        self.assertEqual(cocoon.nlocals, self.states["nlocals"])
        self.assertEqual(cocoon.stack, self.states["stack"])
        self.assertEqual(cocoon.prev_instr_offset, self.states["prev_instr_offset"])
        self.assertEqual(cocoon.is_leaf, self.states["is_leaf"])

    @patch("pyckpt.generator.spawn_objects")
    def test_spawn(self, mock_spawn_objects):
        mock_spawn_objects.side_effect = lambda x, _: x
        cocoon = GeneratorFrameStatesCocoon(**self.states)
        spawn_ctxs = MagicMock(spec=SpawnContextManager)
        spawned_states = cocoon.spawn(spawn_ctxs)
        self.assertEqual(spawned_states["func"], self.states["func"])
        self.assertEqual(spawned_states["nlocals"], self.states["nlocals"])
        self.assertEqual(spawned_states["stack"], self.states["stack"])
        self.assertEqual(
            spawned_states["prev_instr_offset"], self.states["prev_instr_offset"]
        )
        self.assertEqual(spawned_states["is_leaf"], self.states["is_leaf"])


class TestGeneratorContext(unittest.TestCase):
    def setUp(self):
        self.suspended_generators = [MagicMock()]
        self.executing_generators = [MagicMock()]
        self.context = GeneratorContext(
            suspended_generator_states=[],
            executing_generator_states=[],
            suspended_generator=self.suspended_generators,
            executing_generator=self.executing_generators,
        )

    @patch("pyckpt.generator.snapshot_generator")
    @patch("pyckpt.generator.snapshot_generator_frame")
    @patch("pyckpt.generator.GeneratorFrameStatesCocoon.snapshot_from_states")
    def test_snapshot(
        self,
        mock_snapshot_from_states,
        mock_snapshot_generator_frame,
        mock_snapshot_generator,
    ):
        gen_states = {
            "gi_code": "code",
            "gi_name": "name",
            "gi_qualname": "qualname",
        }
        snapshot_frame_states = MagicMock()
        mock_snapshot_generator.return_value = gen_states
        mock_snapshot_generator_frame.return_value = {"stack": [], "nlocals": []}
        mock_snapshot_from_states.return_value = snapshot_frame_states
        cocoon = self.context.snapshot(None)

        self.assertIsInstance(cocoon, CRContextCocoon)

    @patch("pyckpt.generator.interpreter.make_new_generator")
    def test_spawn(self, mock_make_new_generator):
        mock_make_new_generator.return_value = MagicMock()
        spawn_ctxs = MagicMock(spec=SpawnContextManager)
        self.context.suspended_generator_states = [
            (
                1,
                (
                    {"gi_code": "code", "gi_name": "name", "gi_qualname": "qualname"},
                    None,
                ),
            )
        ]
        self.context.spawn(spawn_ctxs)
        mock_make_new_generator.assert_called_once_with(
            self.context.suspended_generator_states[0][1][0]["gi_code"],
            self.context.suspended_generator_states[0][1][0]["gi_name"],
            self.context.suspended_generator_states[0][1][0]["gi_qualname"],
        )
        new_generator = mock_make_new_generator.return_value
        spawn_ctxs.register_object.assert_called_once_with(
            self.context.suspended_generator_states[0][0],
            new_generator,
        )

    @patch("pyckpt.generator.interpreter.setup_generator")
    def test_spawn_epilog(self, mock_setup_generator):
        spawn_ctxs = MagicMock(spec=SpawnContextManager)
        frame_states_mock = MagicMock()
        frame_states_mock.spawn.return_value = {}
        self.context.suspended_generator_states = [
            (
                1,
                (
                    {"gi_code": "code", "gi_name": "name", "gi_qualname": "qualname"},
                    frame_states_mock,
                ),
            )
        ]
        self.context.spawn_epilog(spawn_ctxs)
        mock_setup_generator.assert_called_once()


def test_snapshot_with_local_generator():
    def generator_function():
        yield 41
        yield 42
        return 43

    def test_function_with_generator():
        local_gen = generator_function()
        next(local_gen)  # Advance the generator to its first yield
        ctxs = SnapshotContextManager()
        gen_ctx = GeneratorContext.create_context()
        ctxs.register_context(gen_ctx)
        frame = FrameCocoon.snapshot_from_frame(
            inspect.currentframe(),
            is_leaf=False,
            stack_analyzer=analyze_stack_top,
            contexts=ctxs,
        )
        if frame:
            frame = frame.clone()
            return frame, ctxs.snapshot_contexts()

        return local_gen

    cocoon, contexts = test_function_with_generator()
    # Ensure the cocoon is correctly created
    assert isinstance(cocoon, FrameCocoon)
    spawn_ctxs = SpawnContextManager.build_from_context_snapshot(contexts)
    live_frame = cocoon.spawn(spawn_ctxs)
    spawn_ctxs.epilogue()

    assert isinstance(live_frame, LiveFunctionFrame)

    # Evaluate the generator frame
    gen, exc_states = live_frame._evaluate(None, None)
    assert exc_states is None
    assert isinstance(gen, Generator)
    assert next(gen) == 42

    with pytest.raises(StopIteration) as e:
        next(gen)

    assert e.value.value == 43
