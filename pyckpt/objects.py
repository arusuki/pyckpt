from abc import ABC, abstractmethod
from types import FrameType
from typing import IO, Any, Callable, Dict, List, Optional, Set, Type

import dill

SnapshotMethod = Callable[[Any, "ObjectStates"], Any]
None_PID = 0


class ObjectSnapshotManager(ABC):
    def __init__(self, object_ids: Optional[Set[int]] = None):
        self._ids = object_ids if object_ids else set()
        self._type = type(self)

    @abstractmethod
    def setup_object(self, obj: Any, state: Any): ...

    @abstractmethod
    def snapshot_object(self, obj: Any) -> Any: ...

    @abstractmethod
    def make_new_object(self) -> Any: ...

    @abstractmethod
    def get_snapshot_type(self) -> Type: ...

    def get_snapshot_method(self) -> SnapshotMethod:
        return self._snapshot

    def _snapshot(self, obj: Any, states: "ObjectStates"):
        object_id = id(obj)
        self._ids.add(object_id)
        state = states.get_state_for_mgr(self._type)
        state[object_id] = self.snapshot_object(obj)
        return object_id

    def setup_objects_on_spawn(self, objects: Dict[int, Any], states: Dict[int, Any]):
        for _id, _state in states.items():
            self.setup_object(objects[_id], _state)

    def get_object_ids(self):
        return self._ids


class ObjectStates:
    def __init__(self, states: Optional[Dict[int, Any]] = None):
        self._states: Dict[int, Any] = states if states else {}

    def get_state_for_mgr(
        self, mgr_type: Type[ObjectSnapshotManager]
    ) -> Dict[int, Any]:
        return self._states[mgr_type]

    def make_state_for_mgr(
        self,
        mgr_type: Type[ObjectSnapshotManager],
        states: Optional[Dict[int, Any]] = None,
    ):
        if mgr_type in self._states:
            raise ValueError(
                f"object snapshot manager of type {mgr_type} already registered"
            )

        self._states[mgr_type] = states if states else {}


def snapshot_as_none(_obj: Any, _states: ObjectStates):
    return None_PID


def new_snapshot_registry() -> Dict[Type, SnapshotMethod]:
    return {
        FrameType: snapshot_as_none,  # TODO: support frame types
        ObjectSnapshotManager: snapshot_as_none,
    }


def make_snapshot_registry(
    managers: List[ObjectSnapshotManager],
) -> Dict[Type, SnapshotMethod]:
    registry = new_snapshot_registry()
    for mgr in managers:
        registry[mgr.get_snapshot_type()] = mgr.get_snapshot_method()
    return registry


class Pickler(dill.Pickler):
    def __init__(self, managers: List[ObjectSnapshotManager], file, *args, **kwds):
        super().__init__(file, *args, **kwds)
        self._states = ObjectStates()
        self._registry = make_snapshot_registry(managers)

        for mgr in managers:
            self._states.make_state_for_mgr(type(mgr))

    def persistent_id(self, obj):
        obj_type = type(obj)
        if obj_type not in self._registry:
            return super().persistent_id(obj)
        return self._registry[obj_type](obj, self._states)

    def get_states(self):
        return self._states._states


class Unpickler(dill.Unpickler):
    def __init__(self, file, objects: Dict[int, Any], *args, **kwds):
        super().__init__(file=file, *args, **kwds)
        self._objects = objects

    def persistent_load(self, pid: int):
        if pid == None_PID:
            return None
        return self._objects[pid]


def dump(
    file: IO[bytes],
    cocoon: Any,
    managers: Optional[List[ObjectSnapshotManager]] = None,
) -> List[ObjectSnapshotManager]:
    pickler = Pickler(file=file, managers=managers)
    pickler.dump(cocoon)
    pickler.dump(pickler.get_states())
    return managers


def load(
    file: IO[bytes],
    managers: Optional[List[ObjectSnapshotManager]] = None,
) -> Any:
    objects = {}
    for mgr in managers:
        for _id in mgr.get_object_ids():
            objects[_id] = mgr.make_new_object()
    unpickler = Unpickler(file, objects=objects)
    cocoon = unpickler.load()
    states: Dict = unpickler.load()
    for mgr in managers:
        mgr.setup_objects_on_spawn(objects, states[type(mgr)])
    return cocoon
