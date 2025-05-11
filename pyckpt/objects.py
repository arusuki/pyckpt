from typing import Any, Callable, Dict, Generator, List, Tuple, Type

SnapshotMethod = Callable[[Any, Dict], Any]
SpawnMethod = Callable[[Any, Dict], Any]
RegistryHook = Tuple[SnapshotMethod, SpawnMethod]


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
            snapshot, spawn = registry[obj_type]
            objects[idx] = ObjectCocoon(snapshot(obj, contexts), spawn)
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
