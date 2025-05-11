from pyckpt import objects


def test_snapshot_objects():
    objs = [1, "2", [3]]

    ret = objects.create_snapshot({}, objs, {})

    for o1, o2 in zip(objs, ret):
        assert o1 is o2


def test_stub_objects():
    def spawn_int2str(a: int, _):
        return f"{a}"

    def snapshot_str2int(s: str, _):
        return objects.ObjectCocoon(int(s), spawn_int2str)

    reg = {str: (snapshot_str2int)}

    objs = ["1", "2", "3"]

    ret = objects.create_snapshot(reg, objs, {})

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

    restored_obj = snapshot.spawn_method(snapshot, spawn_context)
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


def test_snapshot_context_manager_build_states():
    context_manager = objects.SnapshotContextManager()

    def mock_state_builder(states, mgr):
        assert isinstance(mgr, objects.SnapshotContextManager)
        states["mock_key"] = "mock_value"

    def mock_spawn_ctx_builder(ctx, states):
        assert isinstance(ctx, objects.SpawnContextManager)
        ctx["mock_spawn_key"] = "mock_spawn_value"

    context_manager.register_state_builder(mock_state_builder, mock_spawn_ctx_builder)
    states = context_manager.build_states()

    assert "mock_key" in states
    assert states["mock_key"] == "mock_value"
    assert "spawn_ctx_builder" in states
    assert len(states["spawn_ctx_builder"]) == 1
    assert states["spawn_ctx_builder"][0] == mock_spawn_ctx_builder


def test_spawn_context_manager_build_from_snapshot_states():
    states = {
        "spawn_ctx_builder": [
            lambda ctx, _: ctx.update({"key1": "value1"}),
            lambda ctx, _: ctx.update({"key2": "value2"}),
        ]
    }

    spawn_context = objects.SpawnContextManager.build_from_snapshot_states(states)

    assert "key1" in spawn_context
    assert spawn_context["key1"] == "value1"
    assert "key2" in spawn_context
    assert spawn_context["key2"] == "value2"


def test_spawn_by_original_id():
    obj = [1, 2, 3]
    obj_id = id(obj)

    spawn_context = objects.SpawnContextManager()
    spawn_context.register_object(obj_id, obj)

    cocoon = objects.snapshot_by_original_id(obj, None)
    restored_obj = cocoon.spawn_method(cocoon, spawn_context)

    assert restored_obj is obj


def test_create_snapshot_with_registry():
    def spawn_list(length, _):
        return [None] * length

    def snapshot_list(lst, _):
        return objects.ObjectCocoon(len(lst), spawn_list)

    registry = {list: (snapshot_list)}
    objs = [[1, 2, 3], [4, 5], [6]]

    snapshots = objects.create_snapshot(registry, objs, {})

    for original, snapshot in zip(objs, snapshots):
        assert isinstance(snapshot, objects.ObjectCocoon)
        assert snapshot.states == len(original)

    restored_objs = objects.spawn_objects(snapshots, {})
    for original, restored in zip(objs, restored_objs):
        assert len(original) == len(restored)
        assert all(item is None for item in restored)
