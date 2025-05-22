import io
from typing import Any

import dill
import pytest

from pyckpt.objects import (
    None_PID,
    ObjectSnapshotManager,
    ObjectStates,
    Unpickler,
    dump,
    load,
    snapshot_as_none,
)


class _CapturedDict(dict):
    pass


class DummyObjectSnapshotManager(ObjectSnapshotManager):
    def setup_object(self, obj: Any, state: Any):
        obj["state"] = state

    def snapshot_object(self, obj: Any) -> Any:
        return obj.get("state", None)

    def make_new_object(self) -> Any:
        return {}

    def get_snapshot_type(self):
        return _CapturedDict


def test_snapshot_and_restore():
    manager = DummyObjectSnapshotManager()
    obj = _CapturedDict()
    obj["state"] = "test_state"
    states = ObjectStates()
    states.make_state_for_mgr(DummyObjectSnapshotManager)

    # Snapshot the object
    object_id = manager._snapshot(obj, states)
    assert object_id in manager.get_object_ids()
    assert states.get_state_for_mgr(type(manager))[object_id] == "test_state"

    # Restore the object
    new_obj = manager.make_new_object()
    manager.setup_object(new_obj, "test_state")
    assert new_obj["state"] == "test_state"


def test_dump_and_load():
    manager = DummyObjectSnapshotManager()
    obj = _CapturedDict()
    obj["state"] = "test_state"
    managers = [manager]

    # Serialize the object
    buffer = io.BytesIO()
    dump(buffer, obj, managers)

    # Deserialize the object
    buffer.seek(0)
    loaded_obj = load(buffer, managers)
    assert loaded_obj["state"] == "test_state"


def test_make_and_get_state_for_mgr():
    states = ObjectStates()
    manager_type = DummyObjectSnapshotManager

    # Create a state for the manager
    states.make_state_for_mgr(manager_type)
    assert manager_type in states._states

    # Retrieve the state for the manager
    state = states.get_state_for_mgr(manager_type)
    assert state == {}

    # Attempt to create a duplicate state for the manager
    with pytest.raises(ValueError):
        states.make_state_for_mgr(manager_type)


def test_snapshot_as_none():
    states = ObjectStates()
    result = snapshot_as_none({}, states)
    assert result == None_PID


def test_unpickler_persist_none():
    class NonePickler(dill.Pickler):
        def persistent_id(self, obj):
            return 0

    buffer = io.BytesIO()
    NonePickler(file=buffer).dump({})

    buffer.seek(0)
    assert Unpickler(file=buffer, objects={}).load() is None
