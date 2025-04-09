import pytest
from pyckpt import objects
from pyckpt.objects import StubObject


def test_stub_objects_no_stubs():
    objs = [1, "2", [3]]

    ret = objects.stub_objects({}, objs, {})

    for o1, o2 in zip(objs, ret):
        assert o1 is o2


def test_stub_objects():
    def save_str_int(s: str, _):
        return int(s)

    def load_str_int(a: int, _):
        return f"{a}"

    reg = {str: (save_str_int, load_str_int)}

    objs = ["1", "2", "3"]

    ret = objects.stub_objects(reg, objs, {})

    for o1, o2 in zip(objs, ret):
        assert isinstance(o2, StubObject)
        assert isinstance(o2.states, int)
        assert int(o1) == o2.states

    rec = objects.get_real_objects(ret, {})

    for o1, o2 in zip(objs, rec):
        assert o1 == o2


def test_recursive_dict_getitem():
    parent = {"a": 1, "b": 2}
    rd = objects.RecursiveDict(parent)
    rd["c"] = 3

    assert rd["a"] == 1
    assert rd["b"] == 2
    assert rd["c"] == 3


def test_recursive_dict_setitem():
    parent = {"a": 1}
    rd = objects.RecursiveDict(parent)
    rd["b"] = 2

    assert rd["b"] == 2
    assert "b" in rd


def test_recursive_dict_delitem():
    parent = {"a": 1}
    rd = objects.RecursiveDict(parent)
    rd["b"] = 2

    del rd["b"]

    assert "b" not in rd

    with pytest.raises(KeyError):
        del rd["a"]


def test_recursive_dict_contains():
    parent = {"a": 1}
    rd = objects.RecursiveDict(parent)
    rd["b"] = 2

    assert "a" in rd
    assert "b" in rd
    assert "c" not in rd


def test_recursive_dict_get():
    parent = {"a": 1}
    rd = objects.RecursiveDict(parent)
    rd["b"] = 2

    assert rd.get("a") == 1
    assert rd.get("b") == 2
    assert rd.get("c", 3) == 3


def test_recursive_dict_keys():
    parent = {"a": 1}
    rd = objects.RecursiveDict(parent)
    rd["b"] = 2

    keys = rd.keys()
    assert "a" in keys
    assert "b" in keys
    assert len(keys) == 2


def test_recursive_dict_items():
    parent = {"a": 1}
    rd = objects.RecursiveDict(parent)
    rd["b"] = 2

    items = dict(rd.items())
    assert items["a"] == 1
    assert items["b"] == 2
    assert len(items) == 2


def test_recursive_dict_values():
    parent = {"a": 1}
    rd = objects.RecursiveDict(parent)
    rd["b"] = 2

    values = list(rd.values())
    assert 1 in values
    assert 2 in values
    assert len(values) == 2


def test_recursive_dict_iter():
    parent = {"a": 1}
    rd = objects.RecursiveDict(parent)
    rd["b"] = 2

    keys = list(iter(rd))
    assert "a" in keys
    assert "b" in keys
    assert len(keys) == 2


def test_recursive_dict_len():
    parent = {"a": 1}
    rd = objects.RecursiveDict(parent)
    rd["b"] = 2

    assert len(rd) == 2


def test_recursive_dict_nested_getitem():
    parent = {"a": 1}
    child = objects.RecursiveDict(parent)
    grandchild = objects.RecursiveDict(child)

    grandchild["b"] = 2

    assert grandchild["a"] == 1
    assert grandchild["b"] == 2
    assert "a" in grandchild
    assert "b" in grandchild


def test_recursive_dict_nested_setitem():
    parent = {"a": 1}
    child = objects.RecursiveDict(parent)
    grandchild = objects.RecursiveDict(child)

    grandchild["b"] = 2
    child["c"] = 3

    assert grandchild["b"] == 2
    assert grandchild["c"] == 3
    assert grandchild["a"] == 1


def test_recursive_dict_nested_delitem():
    parent = {"a": 1}
    child = objects.RecursiveDict(parent)
    grandchild = objects.RecursiveDict(child)

    grandchild["b"] = 2
    del grandchild["b"]

    assert "b" not in grandchild

    with pytest.raises(KeyError):
        del grandchild["a"]


def test_recursive_dict_nested_contains():
    parent = {"a": 1}
    child = objects.RecursiveDict(parent)
    grandchild = objects.RecursiveDict(child)

    grandchild["b"] = 2

    assert "a" in grandchild
    assert "b" in grandchild
    assert "c" not in grandchild


def test_recursive_dict_nested_keys():
    parent = {"a": 1}
    child = objects.RecursiveDict(parent)
    grandchild = objects.RecursiveDict(child)

    grandchild["b"] = 2
    child["c"] = 3

    keys = grandchild.keys()
    assert "a" in keys
    assert "b" in keys
    assert "c" in keys
    assert len(keys) == 3
