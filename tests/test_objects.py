from pyckpt import objects
from pyckpt.objects import StubObject


def test_stub_objects_no_stubs():

    objs = [
        1, "2", [3]
    ]

    ret = objects.stub_objects({}, objs)

    for o1, o2 in zip(objs, ret):
        assert o1 is o2


def test_stub_objects():

    def save_str_int(s: str):
        return int(s)

    def load_str_int(a: int):
        return f"{a}"

    reg = {
        str: (save_str_int, load_str_int)
    }

    objs = ["1", "2", "3"]

    ret = objects.stub_objects(reg, objs)

    for o1, o2 in zip(objs, ret):
        assert isinstance(o2, StubObject)
        assert isinstance(o2.states, int)
        assert int(o1) == o2.states

    rec = objects.get_real_objects(ret)

    for o1, o2 in zip(objs, rec):
        assert o1 == o2
