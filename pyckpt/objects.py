import copyreg
import multiprocessing
from multiprocessing import Process
from queue import SimpleQueue
from threading import Thread
from types import FrameType, NoneType
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Tuple,
    Type,
    Union,
)

import dill
import numpy
import torch
from torch.serialization import default_restore_location as restore_location
from torch.serialization import location_tag

from pyckpt.interpreter.frame import NullObject, NullObjectType
from pyckpt.interpreter.generator import (
    get_generator_type,
    make_generator,
    snapshot_frame_generator,
    snapshot_generator,
)
from pyckpt.interpreter.objects import snapshot_simple_queue

None_PID = 0

Mapping = Callable[[Any], int]
Registry = Dict[Type, Callable[[Any], Tuple[Callable[[Tuple], Any], Tuple]]]


def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    if isinstance(bytes_str, bytes):
        return bytes_str.decode("ascii")
    return bytes_str


def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


def reduce_as_none(_):
    return (NoneType, ())


def reduce_generator(gen: Generator):
    return (
        make_generator,
        (
            snapshot_generator(gen),
            snapshot_frame_generator(gen),
        ),
    )


def _ret_null_object():
    return NullObject


def reduce_null_object(_obj: NullObjectType):
    return _ret_null_object, ()


def reduce_simple_queue(sq: SimpleQueue):
    def make_simple_queue(lst: List):
        q = SimpleQueue()
        for elem in lst:
            q.put(elem)
        return q

    return make_simple_queue, (snapshot_simple_queue(sq),)


class FakeQueue:
    def __init__(self):
        pass

    def __getattribute__(self, _name):
        raise NotImplementedError("accessing multiprocessing.Queue after respawn")


def reduce_multiprocessing_queue(q: multiprocessing.Queue):
    return FakeQueue, ()


def dispatch_table() -> Registry:
    return {
        FrameType: reduce_as_none,  # TODO: support frame types
        get_generator_type(): reduce_generator,
        NullObjectType: reduce_null_object,
        SimpleQueue: reduce_simple_queue,
        type(multiprocessing.Queue()): reduce_multiprocessing_queue,
    }


PersistType = (Thread, Process)


class TensorStorage(dict[str, torch.storage.UntypedStorage]):
    @staticmethod
    def from_typed_storages(storages: dict[str, torch.storage.TypedStorage]):
        tensor_store = TensorStorage()
        tensor_store.update(
            (key, store._untyped_storage) for key, store in storages.items()
        )
        return tensor_store

    def to_typed_storages(self):
        return {
            key: torch.TypedStorage(
                wrap_storage=storage, dtype=torch.uint8, _internal=True
            )
            for key, storage in self.items()
        }

    def __reduce__(self):
        return (TensorStorage.from_typed_storages, (self.to_typed_storages(),))


class PersistedObjects:
    def __init__(self, tensor_storage: TensorStorage):
        self._persisted: dict[Type, set[int]] = {}
        self._tensors: TensorStorage = tensor_storage
        for persisted_type in PersistType:
            self._persisted[persisted_type] = set()

    def get_tensor_storage(self) -> TensorStorage:
        return self._tensors

    def make_persisted_objects(self) -> dict[int, Any]:
        persisted = {}
        for _type, _ids in self._persisted.items():
            persisted.update((_id, object.__new__(_type)) for _id in _ids)
        return persisted

    def persist_object(self, obj: Any) -> int:
        if isinstance(obj, Process):
            raise NotImplementedError("persist process objects")
        original_id = id(obj)
        self._persisted[type(obj)].add(original_id)
        return original_id


def save_untyped_storages(file: IO[bytes], storages: TensorStorage):
    raise NotImplementedError("do not call this function")


def load_untyped_storages(file: IO[bytes]) -> TensorStorage:
    raise NotImplementedError("do not call this function")


def get_leaf_base(array: numpy.ndarray) -> numpy.ndarray | torch.Tensor:
    while isinstance(array, numpy.ndarray) and array.base is not None:
        array = array.base
    return array


def _create_array(t: torch.Tensor):
    return t.numpy()


def save_numpy_with_tensor_base(
    pickler: "Pickler", np_array: numpy.ndarray, base: torch.Tensor
):
    if np_array.__array_interface__["data"][0] != base.untyped_storage().data_ptr():
        raise NotImplementedError("numpy&tensor shared storage with offset")
    pickler.save_reduce(_create_array, (base,), obj=np_array)


class StorageID(Tuple): ...


class BuiltinMethodID(str): ...


BUILTIN_METHODS_TO_ID: dict[Callable, BuiltinMethodID] = {}
ID_TO_BUILTIN_METHODS: dict[BuiltinMethodID, Callable] = {}


def register_builtin(function: Callable):
    function_id = BuiltinMethodID(function.__name__)
    if function_id in ID_TO_BUILTIN_METHODS:
        raise RuntimeError("duplicated builtin methods")
    ID_TO_BUILTIN_METHODS[function_id] = function
    BUILTIN_METHODS_TO_ID[function] = function_id
    return function


class Pickler(dill.Pickler):
    def __init__(self, file, builtin_methods=set(), *args, **kwds):
        super().__init__(file, *args, **kwds)
        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table.update(dispatch_table())
        self.serialized_storages: TensorStorage = TensorStorage()
        self.storage_dtypes: dict[int, torch.dtype] = {}
        self.id_map: dict[int, str] = {}

        self._persisted = PersistedObjects(self.serialized_storages)
        self._builtin_methods = builtin_methods

    def persistent_id(self, obj):
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            return self._persistent_id_storage(obj)
        if obj.__hash__ is not None:
            try:
                obj_id = BUILTIN_METHODS_TO_ID.get(obj, None)
                if obj_id is not None:
                    return obj_id
            except TypeError:
                # some class has __hash__ method but raises TypeError at runtime
                return None
        obj_type = type(obj)
        if obj_type in PersistType:
            return self._persisted.persist_object(obj)
        return None

    def consume_persisted(self) -> PersistedObjects:
        self.serialized_storages = TensorStorage()
        self._persisted, persisted = (
            PersistedObjects(self.serialized_storages),
            self._persisted,
        )
        self.storage_dtypes.clear()
        self.id_map.clear()
        return persisted

    def save(self, obj, save_persistent_id=True):
        if not isinstance(obj, numpy.ndarray):
            return super().save(obj, save_persistent_id)
        base = get_leaf_base(obj)
        if not torch.is_tensor(base):
            return super().save(obj, save_persistent_id)
        assert str(base.device) == "cpu"
        return save_numpy_with_tensor_base(self, obj, base)

    def _persistent_id_storage(self, obj: torch.storage.TypedStorage):
        """
        Persist tensor storage, adopted from\n
        https://github.com/pytorch/pytorch/torch/serialization.py
        """
        if isinstance(obj, torch.storage.TypedStorage):
            storage = obj._untyped_storage
            storage_dtype = obj.dtype
            storage_type_str = obj._pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage_numel = obj._size()

        else:
            storage = obj
            storage_dtype = torch.uint8
            storage_type = normalize_storage_type(type(obj))
            storage_numel = storage.nbytes()

        if str(storage.device) != "meta" and storage.data_ptr() != 0:
            if storage.data_ptr() in self.storage_dtypes:
                if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
                    raise RuntimeError(
                        "Cannot save multiple tensors or storages that "
                        "view the same data as different types"
                    )
            else:
                self.storage_dtypes[storage.data_ptr()] = storage_dtype

        storage_key = self.id_map.setdefault(storage._cdata, str(len(self.id_map)))
        if hasattr(obj, "_fake_device") and obj._fake_device is not None:
            location = str(obj._fake_device)
        else:
            location = location_tag(storage)
        self.serialized_storages[storage_key] = storage
        return StorageID(
            ("storage", storage_type, storage_key, location, storage_numel)
        )


class Unpickler(dill.Unpickler):
    def __init__(
        self,
        file,
        persisted: PersistedObjects,
        *args,
        **kwds,
    ):
        super().__init__(file, *args, **kwds)
        self.loaded_storages: TensorStorage = persisted.get_tensor_storage()
        self._typed_storages: dict[str, torch.storage.TypedStorage] = {}
        self._loaded_objects = persisted.make_persisted_objects()

    def persistent_load(self, pid: int | StorageID):
        if isinstance(pid, StorageID):
            return self._persistent_load_storage(pid)
        elif isinstance(pid, BuiltinMethodID):
            return ID_TO_BUILTIN_METHODS[pid]
        return self._loaded_objects[pid]

    def get_loaded_objects(self) -> Dict[int, Any]:
        return self._loaded_objects

    def _persistent_load_storage(self, pid: StorageID):
        typename = _maybe_decode_ascii(pid[0])
        assert typename == "storage", (
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        )
        storage_type, storage_key, location = pid[1], pid[2], pid[3]
        if storage_key in self._typed_storages:
            return self._typed_storages[storage_key]

        assert storage_key in self.loaded_storages
        storage = self.loaded_storages[storage_key]

        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type._dtype

        if torch._guards.detect_fake_mode(None) is None:
            wrap_storage = restore_location(storage, location)
        else:
            storage._fake_device = location
            wrap_storage = storage

        typed_storage = torch.TypedStorage(
            wrap_storage=wrap_storage, dtype=dtype, _internal=True
        )
        self._typed_storages[storage_key] = typed_storage
        return typed_storage
