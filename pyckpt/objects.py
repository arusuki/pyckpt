import copyreg
from io import BytesIO
from types import FrameType, NoneType
from typing import IO, Any, Callable, Dict, Generator, Optional, Tuple, Type

import dill

from pyckpt.analyzer import analyze_stack_top
from pyckpt.interpreter.frame import NullObject, NullObjectType
from pyckpt.interpreter.generator import (
    get_generator_type,
    make_generator,
    snapshot_generator,
    snapshot_generator_frame,
)

None_PID = 0

Mapping = Callable[[Any], int]
Registry = Dict[Type, Callable[[Any], Tuple[Callable[[Tuple], Any], Tuple]]]


def reduce_as_none(_):
    return (NoneType, ())


def reduce_generator(gen: Generator):
    return (
        make_generator,
        (
            snapshot_generator(gen),
            snapshot_generator_frame(gen, analyze_stack_top),
        ),
    )


def _ret_null_object():
    return NullObject


def reduce_null_object(_obj: NullObjectType):
    return _ret_null_object, ()


def new_snapshot_registry() -> Registry:
    return {
        FrameType: reduce_as_none,  # TODO: support frame types
        get_generator_type(): reduce_generator,
        NullObjectType: reduce_null_object,
    }


class Pickler(dill.Pickler):
    def __init__(self, file, persistent_mapping: Dict[Type, Mapping], *args, **kwds):
        super().__init__(file, *args, **kwds)
        self._pm = persistent_mapping if persistent_mapping else {}

    def persistent_id(self, obj):
        obj_type = type(obj)
        if obj_type in self._pm:
            return self._pm[obj_type](obj)
        return None


def dump(
    file: IO[bytes],
    cocoon: Any,
    reduce_registry: Optional[Registry] = None,
    persist_mapping: Optional[Dict[Type, Mapping]] = None,
):
    pickler = Pickler(file=file, persistent_mapping=persist_mapping)
    dispatch_table = copyreg.dispatch_table.copy()
    registry = new_snapshot_registry()
    if reduce_registry:
        registry.update(reduce_registry)
    for _type, _entry in registry.items():
        dispatch_table[_type] = _entry
    pickler.dispatch_table = dispatch_table
    pickler.dump(cocoon)


class Unpickler(dill.Unpickler):
    def __init__(self, file, objects, *args, **kwds):
        super().__init__(file, *args, **kwds)
        self._objects = objects

    def persistent_load(self, pid: int):
        return self._objects[pid]


def load(file: IO[bytes], objects: Dict) -> Any:
    unpickler = Unpickler(file, objects=objects)
    return unpickler.load()


def copy(
    cocoon: Any,
    objects: Optional[Dict] = None,
    reduce_registry: Optional[Registry] = None,
    persist_mapping: Optional[Dict[Type, Mapping]] = None,
):
    buffer = BytesIO()
    dump(buffer, cocoon, reduce_registry, persist_mapping)
    buffer.seek(0)
    objects = objects if objects else {}
    return load(buffer, objects)
