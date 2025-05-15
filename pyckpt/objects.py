from abc import ABC, abstractmethod
from types import FrameType, NoneType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from attr import dataclass

SnapshotMethod = Callable[[Any, "SnapshotContextManager"], "ObjectCocoon"]
SpawnMethod = Callable[[Any, "SpawnContextManager"], Any]
RegistryHook = Tuple[SnapshotMethod]
ContextStateBuilder = Callable[[Dict, "SnapshotContextManager"], NoneType]
SpawnContextBuilder = Callable[["SpawnContextManager", Dict], NoneType]


@dataclass(slots=True)
class CRContextCocoon:
    spawn_method: Callable[[Any], "CheckpointRestoreContext"]
    states: Any

    def spawn(self) -> "CheckpointRestoreContext":
        return self.spawn_method(self.states)


class CheckpointRestoreContext(ABC):
    @abstractmethod
    def snapshot(self, snapshot_ctxs: "SnapshotContextManager") -> CRContextCocoon:
        pass

    @abstractmethod
    def spawn(self, spawn_ctxs: "SpawnContextManager") -> "CheckpointRestoreContext":
        pass

    def spawn_epilog(self, spawn_ctxs: "SpawnContextManager"):
        return

    def register_snapshot_method(
        self, snapshot_ctxs: "SnapshotContextManager"
    ) -> Optional[SpawnMethod]:
        return


class ContextManager:
    def __init__(self, contexts: Optional[Dict]):
        if contexts is None:
            contexts = {}
        self._contexts: Dict[Type, CheckpointRestoreContext] = contexts

    C = TypeVar("C")

    def __getitem__(self, context_type: C) -> C:
        return self.get_context(context_type)

    def get_context(self, context_type: C) -> C:
        """
        Retrieve a context of the specified type.
        """
        return self._contexts[context_type]

    def register_context(self, context: CheckpointRestoreContext):
        """
        Register a context using its type as the key.
        """
        context_type = type(context)
        if context_type in self._contexts:
            raise ValueError(
                f"context of type {context_type} has already been registered"
            )
        self._contexts[type(context)] = context


class SnapshotContextManager(ContextManager):
    def __init__(
        self,
        contexts: Optional[Dict] = None,
        registry: Optional[Dict] = None,
    ):
        if registry is None:
            registry = new_snapshot_registry()
        self._registry = registry
        super().__init__(contexts)

    def snapshot_contexts(self) -> List[CRContextCocoon]:
        return [ctx.snapshot(self) for ctx in self._contexts.values()]

    def snapshot_registry(self):
        return self._registry

    def register_snapshot_method(self, object_type: Type, snapshot: SnapshotMethod):
        self._registry[object_type] = snapshot

    def registry(self):
        return self._registry

    def register_context(self, context: CheckpointRestoreContext):
        context.register_snapshot_method(self)
        return super().register_context(context)


class SpawnContextManager(ContextManager):
    def __init__(self, contexts: Optional[Dict] = None):
        super().__init__(contexts)
        self._object_pool: Dict[int, Any] = {}
        for ctx in self._contexts.values():
            ctx.spawn(self)

    @staticmethod
    def build_from_context_snapshot(states: List[CRContextCocoon]):
        ctxs = [cocoon.spawn() for cocoon in states]
        return SpawnContextManager({type(ctx): ctx for ctx in ctxs})

    def register_object(self, original_id, obj):
        self._object_pool[original_id] = obj

    def retrieve_object(self, original_id):
        if original_id not in self._object_pool:
            raise ValueError(
                f"Object with original ID {original_id} not found in the object pool"
            )
        return self._object_pool[original_id]

    def epilogue(self):
        for ctx in self._contexts.values():
            ctx.spawn_epilog(self)


def register_type_snapshot_by_id(
    object_type: Type,
    registry: Dict[Type, RegistryHook],
):
    registry[object_type] = snapshot_by_original_id


class ObjectCocoon:
    def __init__(
        self,
        states: Any,
        spawn_method: SpawnMethod,
    ):
        self.states = states
        self.spawn_method = spawn_method


def snapshot_objects(objects: List[Any], contexts: SnapshotContextManager):
    """
    Create snapshots of objects using the provided registry.

    Args:
        registry: A dictionary mapping object types to their snapshot and spawn methods.
        objects: A list of objects to snapshot.
        contexts: A dictionary of additional context information.

    Returns:
        A new list of objects, where applicable objects are replaced with ObjectCocoon instances.
    """
    objects = objects.copy()
    registry = contexts.registry()
    for idx, obj in enumerate(objects):
        obj_type = type(obj)
        if obj_type in registry:
            snapshot = registry[obj_type]
            objects[idx] = snapshot(obj, contexts)
    return objects


def spawn_objects(objects: List[Any], contexts: SpawnContextManager):
    """
    Restore objects from their snapshots.

    Args:
        objects: A list of objects, some of which may be ObjectCocoon instances.
        contexts: A dictionary of additional context information.

    Returns:
        A new list of objects, where ObjectCocoon instances are replaced with their original objects.
    """
    objects = objects.copy()
    for idx, obj in enumerate(objects):
        if isinstance(obj, ObjectCocoon):
            objects[idx] = obj.spawn_method(obj.states, contexts)
    return objects


def _spawn_by_original_id(obj: int, mgr: SpawnContextManager):
    return mgr.retrieve_object(obj)


def snapshot_by_original_id(obj: Any, _mgr: SnapshotContextManager):
    original_id = id(obj)
    return ObjectCocoon(original_id, _spawn_by_original_id)


def _spawn_none(_obj: Any, _mgr: SpawnContextManager):
    return None


def snapshot_as_none(_obj: Any, _mgr: SnapshotContextManager):
    return ObjectCocoon(None, _spawn_none)


def new_snapshot_registry():
    return {
        FrameType: snapshot_as_none,  # TODO: support frame types
        SnapshotContextManager: snapshot_as_none,
        SpawnContextManager: snapshot_as_none,
    }
