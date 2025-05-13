from types import NoneType
from typing import Any, Callable, Dict, Generator, List, Tuple, Type, TypeVar

SnapshotMethod = Callable[[Any, Dict], "ObjectCocoon"]
SpawnMethod = Callable[[Any, Dict], Any]
RegistryHook = Tuple[SnapshotMethod]
ContextStateBuilder = Callable[[Dict, "SnapshotContextManager"], NoneType]
SpawnContextBuilder = Callable[["SpawnContextManager", Dict], NoneType]


class SnapshotContextManager(dict):
    def __init__(self):
        super().__init__()
        self.state_builder: List[ContextStateBuilder] = []
        self.spawn_ctx_builders = []

    def build_states(self) -> Dict:
        states = {}
        states["spawn_ctx_builder"] = self.spawn_ctx_builders.copy()
        for builder in self.state_builder:
            builder(states, self)
        return states

    def register_state_builder(
        self,
        state_builder: ContextStateBuilder,
        spawn_ctx_builder: SpawnContextBuilder,
    ):
        self.state_builder.append(state_builder)
        self.spawn_ctx_builders.append(spawn_ctx_builder)

    C = TypeVar("C")

    def get_context(self, context_type: C) -> C:
        """
        Retrieve a context of the specified type.
        """
        return self[context_type]

    def register_context(self, context: Any):
        """
        Register a context using its type as the key.
        """
        super().__setitem__(type(context), context)

    def __setitem__(self, _key, _value):
        raise ValueError(
            "directly set snapshot contexts is not allowed, use `register_context()` instead"
        )


class SpawnContextManager(dict):
    def __init__(self):
        self._object_pool: Dict[int, Any] = {}

    def register_object(self, original_id, obj):
        self._object_pool[original_id] = obj

    def retrieve_object(self, original_id):
        if original_id not in self._object_pool:
            raise ValueError(
                f"Object with original ID {original_id} not found in the object pool"
            )
        return self._object_pool[original_id]

    @staticmethod
    def build_from_snapshot_states(states: Dict):
        ctx = SpawnContextManager()
        spawn_builders: List[SpawnContextBuilder] = states["spawn_ctx_builder"]
        for spawn_builder in spawn_builders:
            spawn_builder(ctx, states)
        return ctx


def _spawn_by_original_id(obj: int, mgr: SpawnContextManager):
    return mgr.retrieve_object(obj)


def snapshot_by_original_id(obj: Any, _mgr: SnapshotContextManager):
    original_id = id(obj)
    return ObjectCocoon(original_id, _spawn_by_original_id)


def register_type_snapshot_by_id(
    object_type: Type,
    registry: Dict[Type, RegistryHook],
):
    registry[object_type] = snapshot_by_original_id


class NullObjectType:
    pass


NullObject = NullObjectType()


class ObjectCocoon:
    def __init__(
        self,
        states: Any,
        spawn_method: SpawnMethod,
    ):
        self.states = states
        self.spawn_method = spawn_method


def create_snapshot(
    registry: Dict[Type, RegistryHook],
    objects: List[Any],
    contexts: Dict,
):
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
    for idx, obj in enumerate(objects):
        if isinstance(obj, Generator):
            raise NotImplementedError("save generator type")
        obj_type = type(obj)
        if obj_type in registry:
            snapshot = registry[obj_type]
            objects[idx] = snapshot(obj, contexts)
    return objects


def spawn_objects(objects: List[Any], contexts: Dict):
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
