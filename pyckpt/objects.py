from typing import Any, Callable, Dict, Generator, List, Tuple, Type

LoadHook = Callable[[Any], Any]
StoreHook = Callable[[Any, Dict], Any]
RegistryHook = Tuple[LoadHook, StoreHook]


class NullObjectType:
    pass


NullObject = NullObjectType()


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
