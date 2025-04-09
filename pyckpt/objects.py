from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type

LoadHook = Callable[[Any], Any]
StoreHook = Callable[[Any, Dict], Any]
RegistryHook = Tuple[LoadHook, StoreHook]


class NullObjectType:
    pass


NullObject = NullObjectType()


class RecursiveDict:
    def __init__(self, parent: Optional[Dict]):
        if parent is None:
            parent = {}
        self._parent = parent
        self._local = {}

    def __getitem__(self, key: Any) -> Any:
        if key in self._local:
            return self._local[key]
        return self._parent[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._local[key] = value

    def __delitem__(self, key: Any) -> None:
        if key in self._local:
            del self._local[key]
        else:
            raise KeyError(f"Key '{key}' not found in local dictionary.")

    def __contains__(self, key: Any) -> bool:
        return key in self._local or key in self._parent

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self._local:
            return self._local[key]
        return self._parent.get(key, default)

    def keys(self):
        return set(self._local.keys()).union(self._parent.keys())

    def items(self):
        combined = {**self._parent, **self._local}
        return combined.items()

    def values(self):
        combined = {**self._parent, **self._local}
        return combined.values()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


class StubObject:
    def __init__(
        self,
        states: Any,
        load_hook: LoadHook,
    ):
        self.load_hook = load_hook
        self.states = states

    def load_client(self, contexts: Dict):
        return self.load_hook(self.states, contexts)


def stub_objects(
    registry: Dict[Type, RegistryHook], objects: List[Any], contexts: Dict
):
    objects = objects.copy()
    for idx, obj in enumerate(objects):
        if isinstance(obj, Generator):
            raise NotImplementedError("save generator type")
        obj_type = type(obj)
        if obj_type in registry:
            store, load = registry[obj_type]
            objects[idx] = StubObject(store(obj, contexts), load)
    return objects


def get_real_objects(objects: List[Any], contexts: Dict):
    objects = objects.copy()
    for idx, obj in enumerate(objects):
        if isinstance(obj, StubObject):
            objects[idx] = obj.load_client(contexts)
    return objects
