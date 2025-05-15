import dataclasses
from dataclasses import dataclass
from itertools import chain
from types import FunctionType
from typing import Any, Dict, Generator, List, Set, Tuple

from pyckpt import interpreter
from pyckpt.analyzer import analyze_stack_top
from pyckpt.interpreter.generator import snapshot_generator, snapshot_generator_frame
from pyckpt.objects import (
    CheckpointRestoreContext,
    CRContextCocoon,
    SnapshotContextManager,
    SpawnContextManager,
    snapshot_as_none,
    snapshot_by_original_id,
    snapshot_objects,
    spawn_objects,
)


@dataclass
class GeneratorFrameStatesCocoon:
    func: FunctionType
    nlocals: List[Any]
    stack: List[Any]
    prev_instr_offset: int
    is_leaf: bool

    @staticmethod
    def snapshot_from_states(states: Dict, snapshot_contexts: SnapshotContextManager):
        stack = snapshot_objects(states["stack"], snapshot_contexts)
        nlocals = snapshot_objects(states["nlocals"], snapshot_contexts)
        return GeneratorFrameStatesCocoon(
            func=states["func"],
            nlocals=nlocals,
            stack=stack,
            prev_instr_offset=states["prev_instr_offset"],
            is_leaf=states["is_leaf"],
        )

    def spawn(self, spawn_ctxs: SpawnContextManager):
        states = dataclasses.asdict(self)
        states["stack"] = spawn_objects(states["stack"], spawn_ctxs)
        states["nlocals"] = spawn_objects(states["nlocals"], spawn_ctxs)
        return states


GeneratorStates = Dict
FrameStates = Dict
Generators = Tuple[int, Tuple[GeneratorStates, FrameStates]]


@dataclass
class GeneratorContext(CheckpointRestoreContext):
    # used for snapshot
    suspended_generator: Set[Generator]
    executing_generator: Set[Generator]
    # used for spawn
    suspended_generator_states: List[Generators]
    executing_generator_states: List[Generators]

    @staticmethod
    def create_context():
        return GeneratorContext(
            suspended_generator=set(),
            executing_generator=set(),
            suspended_generator_states=[],
            executing_generator_states=[],
        )

    @staticmethod
    def snapshot_generator(gen: Generator, ctxs: SnapshotContextManager):
        ctx = ctxs.get_context(GeneratorContext)
        if interpreter.is_suspended(gen):
            ctx.suspended_generator.add(gen)
        elif interpreter.is_executing(gen):
            ctx.executing_generator.add(gen)
        else:
            raise NotImplementedError(
                "snapshot not-started|cleared generator is not implemented yet"
            )
        return snapshot_by_original_id(gen, ctxs)

    def register_snapshot_method(self, snapshot_ctxs: "SnapshotContextManager"):
        snapshot_ctxs.register_snapshot_method(
            interpreter.get_generator_type(), GeneratorContext.snapshot_generator
        )
        snapshot_ctxs.register_snapshot_method(GeneratorContext, snapshot_as_none)

    @staticmethod
    def cocoon_spawn(states: Tuple[Generators, Generators]):
        return GeneratorContext(
            suspended_generator=None,
            executing_generator=None,
            suspended_generator_states=states[0],
            executing_generator_states=states[1],
        )

    def snapshot(self, snapshot_ctxs: SnapshotContextManager) -> CRContextCocoon:
        suspended_generator_states = []
        for gen in self.suspended_generator:
            gen_state = snapshot_generator(gen)
            gen_frame_states = snapshot_generator_frame(gen, analyze_stack_top)
            gen_frame_states = GeneratorFrameStatesCocoon.snapshot_from_states(
                gen_frame_states, snapshot_ctxs
            )
            suspended_generator_states.append((id(gen), (gen_state, gen_frame_states)))

        executing_generator_states = []
        for gen in self.executing_generator:
            gen_state = snapshot_generator(gen)
            (executing_generator_states.append((id(gen), (gen_state, None))),)

        states = (suspended_generator_states, executing_generator_states)
        return CRContextCocoon(GeneratorContext.cocoon_spawn, states)

    def spawn(self, spawn_ctxs):
        for original_id, (gen_states, _) in chain(
            self.suspended_generator_states, self.executing_generator_states
        ):
            gen = interpreter.make_new_generator(
                gen_states["gi_code"],
                gen_states["gi_name"],
                gen_states["gi_qualname"],
            )
            spawn_ctxs.register_object(original_id, gen)

    def spawn_epilog(self, spawn_ctxs: "SpawnContextManager"):
        for original_id, (
            gen_states,
            frame_states,
        ) in self.suspended_generator_states:
            gen = spawn_ctxs.retrieve_object(original_id)
            frame_states = frame_states.spawn(spawn_ctxs)
            interpreter.setup_generator(gen, gen_states, frame_states)
