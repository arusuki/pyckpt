from typing import Any, Callable, Dict, List, Type

HookType = Callable[[Any], Any]


class StubObject:

    def __init__(
        self,
        client: Any,
        store_hook: HookType,
        load_hook: HookType,
    ):
        self.load_hook = load_hook
        self.states = store_hook(client)

    def load_client(self):
        return self.load_hook(self.states)


def stub_objects(
    registry: Dict[Type, HookType],
    objects: List[Any],
):
    objects = objects.copy()
    for idx, obj in enumerate(objects):
        obj_type = type(obj)
        if obj_type in registry:
            store, load = registry[obj_type]
            objects[idx] = StubObject(obj, store, load)
    return objects


def get_real_objects(objects: List[Any]):
    objects = objects.copy()
    for idx, obj in enumerate(objects):
        if isinstance(obj, StubObject):
            objects[idx] = obj.load_client()
    return objects
