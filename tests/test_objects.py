from pyckpt import objects


def test_snapshot_objects():
    objs = [1, "2", [3]]

    ret = objects.create_snapshot({}, objs, {})

    for o1, o2 in zip(objs, ret):
        assert o1 is o2


def test_stub_objects():
    def snapshot_str2int(s: str, _):
        return int(s)

    def spawn_int2str(a: int, _):
        return f"{a}"

    reg = {str: (snapshot_str2int, spawn_int2str)}

    objs = ["1", "2", "3"]

    ret = objects.create_snapshot(reg, objs, {})

    for o1, o2 in zip(objs, ret):
        assert isinstance(o2, objects.ObjectCocoon)
        assert isinstance(o2.states, int)
        assert int(o1) == o2.states

    rec = objects.spawn_objects(ret, {})

    for o1, o2 in zip(objs, rec):
        assert o1 == o2
