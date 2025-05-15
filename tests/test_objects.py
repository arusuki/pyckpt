from unittest.mock import Mock

from pyckpt import objects
from pyckpt.objects import (
    CheckpointRestoreContext,
    CRContextCocoon,
    SnapshotContextManager,
)


def test_snapshot_objects():
    objs = [1, "2", [3]]
    ctxs = SnapshotContextManager()
    ret = objects.snapshot_objects(objs, ctxs)

    for o1, o2 in zip(objs, ret):
        assert o1 is o2


def test_stub_objects():
    def spawn_int2str(a: int, _):
        return f"{a}"

    def snapshot_str2int(s: str, _):
        return objects.ObjectCocoon(int(s), spawn_int2str)

    reg = {str: (snapshot_str2int)}

    objs = ["1", "2", "3"]

    ctxs = SnapshotContextManager(registry=reg)
    ret = objects.snapshot_objects(objs, ctxs)

    for o1, o2 in zip(objs, ret):
        assert isinstance(o2, objects.ObjectCocoon)
        assert isinstance(o2.states, int)
        assert int(o1) == o2.states

    rec = objects.spawn_objects(ret, {})

    for o1, o2 in zip(objs, rec):
        assert o1 == o2


def test_snapshot_by_original_id():
    obj = [1, 2, 3]

    snapshot = objects.snapshot_by_original_id(obj, None)
    assert isinstance(snapshot, objects.ObjectCocoon)
    assert snapshot.states == id(obj)

    spawn_context = objects.SpawnContextManager()
    spawn_context.register_object(id(obj), obj)

    restored_obj = snapshot.spawn_method(snapshot.states, spawn_context)
    assert restored_obj is obj


def test_spawn_context_manager_retrieve_object():
    obj = [1, 2, 3]
    obj_id = id(obj)

    spawn_context = objects.SpawnContextManager()
    spawn_context.register_object(obj_id, obj)

    retrieved_obj = spawn_context.retrieve_object(obj_id)
    assert retrieved_obj is obj

    try:
        spawn_context.retrieve_object(99999)
    except ValueError as e:
        assert str(e) == "Object with original ID 99999 not found in the object pool"


def test_spawn_by_original_id():
    obj = [1, 2, 3]
    obj_id = id(obj)

    spawn_context = objects.SpawnContextManager()
    spawn_context.register_object(obj_id, obj)

    cocoon = objects.snapshot_by_original_id(obj, None)
    restored_obj = cocoon.spawn_method(cocoon.states, spawn_context)

    assert restored_obj is obj


def test_create_snapshot_with_registry():
    def spawn_list(length, _):
        return [None] * length

    def snapshot_list(lst, _):
        return objects.ObjectCocoon(len(lst), spawn_list)

    objs = [[1, 2, 3], [4, 5], [6]]

    registry = {list: (snapshot_list)}
    ctxs = SnapshotContextManager(registry=registry)
    snapshots = objects.snapshot_objects(objs, ctxs)

    for original, snapshot in zip(objs, snapshots):
        assert isinstance(snapshot, objects.ObjectCocoon)
        assert snapshot.states == len(original)

    restored_objs = objects.spawn_objects(snapshots, {})
    for original, restored in zip(objs, restored_objs):
        assert len(original) == len(restored)
        assert all(item is None for item in restored)


def _create_cr_context_mock(states=None):
    mock_context = Mock(spec=CheckpointRestoreContext)

    def snapshot_behavior(snapshot_ctxs):
        return CRContextCocoon(spawn_method=lambda states: mock_context, states=states)

    mock_context.snapshot.side_effect = snapshot_behavior
    return mock_context


def test_snapshot_context_manager_snapshot_contexts():
    s = "hello"
    mock_context = _create_cr_context_mock(s)
    snapshot_manager = objects.SnapshotContextManager(
        {type(mock_context): mock_context}
    )

    snapshots = snapshot_manager.snapshot_contexts()
    assert len(snapshots) == 1
    assert isinstance(snapshots[0], objects.CRContextCocoon)
    assert snapshots[0].states is s


def test_spawn_context_manager_build_from_context_snapshot():
    mock_context = _create_cr_context_mock()
    snapshot_manager = objects.SnapshotContextManager(
        {type(mock_context): mock_context}
    )
    snapshots = snapshot_manager.snapshot_contexts()
    spawn_manager = objects.SpawnContextManager.build_from_context_snapshot(snapshots)

    assert isinstance(spawn_manager, objects.SpawnContextManager)
    assert type(mock_context) in spawn_manager._contexts
    mock_context.spawn.assert_called_once_with(spawn_manager)


def test_spawn_context_manager_register_and_retrieve_object():
    obj = [1, 2, 3]
    obj_id = id(obj)

    spawn_manager = objects.SpawnContextManager()
    spawn_manager.register_object(obj_id, obj)

    retrieved_obj = spawn_manager.retrieve_object(obj_id)
    assert retrieved_obj is obj

    try:
        spawn_manager.retrieve_object(99999)
    except ValueError as e:
        assert str(e) == "Object with original ID 99999 not found in the object pool"


def test_spawn_context_manager_epilogue():
    mock_context = _create_cr_context_mock()
    spawn_manager = objects.SpawnContextManager({type(mock_context): mock_context})
    spawn_manager.epilogue()

    mock_context.spawn_epilog.assert_called_once()
