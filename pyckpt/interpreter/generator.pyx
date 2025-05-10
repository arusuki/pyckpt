# cython: embedsignature=True, embedsignature.format=python
import inspect
import sys
from types import TracebackType

from cpython.ref cimport Py_INCREF, PyTypeObject
from typing import Dict, Generator, Type
from pyckpt.interpreter.cpython cimport *
from pyckpt.interpreter.cpython cimport (
    _Py_CODEUNIT,
    _PyEval_EvalFrameDefault,
)
from pyckpt.interpreter.frame cimport init_locals_and_stack, _snapshot_frame
from pyckpt.interpreter.frame import NullObject, fetch_exception, restore_exception


if (3, 11) <= sys.version_info <= (3, 12):
    from pyckpt.interpreter.cpython cimport PyInterpreterFrame_311 as _PyInterpreterFrame
    from pyckpt.interpreter.cpython cimport PyFrame_InitializeSpecials_311 as _PyFrame_InitializeSpecials
    from pyckpt.interpreter.cpython cimport get_frame_cleared_311 as get_frame_cleared
    from pyckpt.interpreter.cpython cimport get_frame_created_311 as get_frame_created
    from pyckpt.interpreter.cpython cimport get_frame_suspended_311 as get_frame_suspended
    from pyckpt.interpreter.cpython cimport get_frame_executing_311 as get_frame_executing
    from pyckpt.interpreter.cpython cimport get_frame_owned_by_generator_311 as get_frame_owned_by_generator
else:
    raise SystemError(f"unsupported python version: {sys.version_info}")

cdef extern from "Python.h":
    """
    # define New_Gen(slots) PyObject_GC_NewVar(PyGenObject, &PyGen_Type, slots)

    # define check_exception_StopIteration() (PyErr_ExceptionMatches(PyExc_StopIteration))
    """

    cdef int check_exception_StopIteration()

    ctypedef struct _PyErr_StackItem:
        PyObject* exc_value
        _PyErr_StackItem *previous_item

    ctypedef struct PyGenObject:
        PyCodeObject *gi_code;
        PyObject *gi_weakreflist;
        PyObject *gi_name;
        PyObject *gi_qualname;
        PyObject *gi_origin_or_finalizer;
        char gi_hooks_inited;
        char gi_closed;
        char gi_running_async;
        _PyErr_StackItem gi_exc_state;
        int gi_frame_state;
        _PyInterpreterFrame *gi_iframe;

    cdef PyGenObject* New_Gen(int slots)

    cdef PyTypeObject PyGen_Type

cdef PyGenObject* new_generator(PyFunctionObject *func):
    cdef PyCodeObject* code = <PyCodeObject*> func.func_code
    cdef int slots = code.co_nlocalsplus + code.co_stacksize;
    cdef PyGenObject* gen = New_Gen(slots)
    if gen == NULL:
        return NULL
    if func.func_name == NULL or func.func_qualname == NULL:
        return NULL
    gen.gi_frame_state = get_frame_cleared()
    gen.gi_code = code
    gen.gi_exc_state.previous_item = NULL
    gen.gi_exc_state.exc_value = NULL
    gen.gi_weakreflist = NULL
    gen.gi_name = func.func_name
    gen.gi_qualname = func.func_qualname
    Py_INCREF(<object> gen.gi_code)
    Py_INCREF(<object> gen.gi_name)
    Py_INCREF(<object> gen.gi_qualname)
    PyObject_GC_Track(<PyObject*> gen)
    return gen


cdef inline int no_exception(PyObject* exc_value):
    return exc_value == NULL or exc_value == Py_None

def snapshot_generator(generator: Generator):
    cdef PyGenObject* gen = <PyGenObject*> generator

    Py_INCREF(<object> gen.gi_code)
    Py_INCREF(<object> gen.gi_name)
    Py_INCREF(<object> gen.gi_qualname)
    Py_INCREF(<object> gen.gi_frame_state)

    cdef PyObject* exc_value = gen.gi_exc_state.exc_value
    exception = NullObject
    if exc_value:
        exception = <object> exc_value
        Py_INCREF(exception)

    return {
        "gi_code"        : <object> gen.gi_code,
        "gi_name"        : <object> gen.gi_name,
        "gi_qualname"    : <object> gen.gi_qualname,
        "gi_frame_state" : <int>    gen.gi_frame_state,
        "suspended": gen.gi_frame_state == get_frame_suspended(),
        "exception": exception,
    }

cpdef snapshot_generator_frame(object generator, object analyzer):
    cdef PyGenObject* gen = <PyGenObject*> generator
    if gen.gi_frame_state != get_frame_suspended():
        raise ValueError("snapshot non-suspended generator is not supported")
    return _snapshot_frame(gen.gi_iframe, False, analyzer)

def make_generator(gen_states: Dict, frame_states: Dict):
    cdef PyFunctionObject* func = <PyFunctionObject*> frame_states["func"]
    # N.B. The generator somehow should own a strong reference to the funcobject.
    # This is different from frame evaluation apis where the caller owns the reference.
    # Logically, the generator should own everything it could `touch`,and it can touch
    # the funcobject through the _PyInterpreterFrame C struct (which is not a PyObject!)
    # So just make the GC happy :)
    Py_INCREF(<object> func)
    cdef PyGenObject* gen = new_generator(func)
    cdef _PyInterpreterFrame* frame = gen.gi_iframe
    cdef PyCodeObject* code = <PyCodeObject*> func.func_code

    exception = gen_states["exception"]
    if exception is not NullObject:
        gen.gi_exc_state.exc_value = <PyObject*> (exception)
        Py_INCREF(exception)

    nlocals = frame_states["nlocals"]
    prev_instr_offset = frame_states["prev_instr_offset"]
    _PyFrame_InitializeSpecials(
        frame, func, <PyObject*> nlocals, code.co_nlocalsplus
    )
    frame.prev_instr = &(<_Py_CODEUNIT*>(code.co_code_adaptive))[prev_instr_offset]
    frame.owner = get_frame_owned_by_generator()
    gen.gi_frame_state = gen_states["gi_frame_state"]
    stack = frame_states["stack"]
    init_locals_and_stack(frame, nlocals, stack, code.co_nlocalsplus)
    if gen == NULL:
        raise RuntimeError("failed to make new generator")
    return <object> gen

def is_suspended(generator: Generator):
    return (<PyGenObject*> generator).gi_frame_state == get_frame_suspended()

def is_executing(generator: Generator):
    return (<PyGenObject*> generator).gi_frame_state == get_frame_executing()

def get_generator_type() -> Type:
    # `type` objects are not dynamically allocated
    # hence no incref here
    return <object> &PyGen_Type

cdef _push_generator_exception(PyGenObject* gen):
    cdef PyThreadState* tstate = PyThreadState_GET()

    gen.gi_exc_state.previous_item = <_PyErr_StackItem*> tstate.exc_info
    tstate.exc_info = &gen.gi_exc_state

cdef _pop_generator_exception(PyGenObject* gen):
    cdef PyThreadState* tstate = PyThreadState_GET()

    tstate.exc_info = gen.gi_exc_state.previous_item
    gen.gi_exc_state.previous_item = NULL


cpdef object resume_generator(object generator, object is_leaf, object ret_val = None, exc_states = None):
    """resume_generator_pop(generator: Generator, is_leaf: bool, ret_val: Any) -> Tuple[Any, Optional[ExceptionStates]]

        Mimic CPython's behavior for generator evaluation
        See gen_send_ex2() in https://github.com/python/cpython/blob/3.11/Objects/genobject.c
    """
    cdef PyGenObject* gen = <PyGenObject*> generator
    if gen.gi_frame_state != get_frame_suspended() \
        and gen.gi_frame_state != get_frame_executing():
        raise ValueError(f"invalid generator: {generator}: frame cleared or not started")
    cdef _PyInterpreterFrame* frame = gen.gi_iframe
    cdef PyThreadState* tstate = PyThreadState_GET()
    frame.previous = <_PyInterpreterFrame*> tstate.cframe.current_frame
    cdef int _is_leaf = <int> is_leaf
    gen.gi_frame_state = get_frame_executing()

    if not is_leaf:
        Py_INCREF(ret_val)
        frame.localsplus[frame.stacktop] = <PyObject*> ret_val;
        frame.stacktop += 1

    cdef int do_exc = 0
    if exc_states is not None:
        if PyErr_Occurred() != NULL:
            _, exc_prev, _ = fetch_exception()
            PyErr_Clear()
            raise RuntimeError("eval a frame when exception has been raised") from exc_prev
        restore_exception(exc_states)
        do_exc = 1
    _push_generator_exception(gen)
    cdef PyObject* result = _PyEval_EvalFrameDefault(tstate, frame, do_exc)
    _pop_generator_exception(gen)
    frame.previous = NULL
    cdef PyObject* p_type  = NULL
    cdef PyObject* p_value = NULL
    cdef PyObject* p_traceback = NULL
    if result != NULL and gen.gi_frame_state == get_frame_suspended():
        return (<object> result, None) # return by `yield`
    # manually clear the interpreter frame owned by generator
    if result == NULL:
        PyErr_Fetch(&p_type, &p_value, &p_traceback)
        assert p_type != NULL and p_value != NULL
        assert not isinstance(<object> p_value, StopIteration)
        tb = None
        if p_traceback != NULL:
            tb = <object> p_traceback
        if clear_generator_frame(gen) != 0:
            raise RuntimeError(f"clear interpreter frame failed, generator: {generator}")
        return (NullObject, (<object> p_type, <object> p_value, tb))
    result_obj = <object> result
    e = StopIteration(result_obj)

    if clear_generator_frame(gen) != 0:
        raise RuntimeError(f"clear interpreter frame failed, generator: {generator}")
    # TODO: add traceback support, here we simply pass `None`
    tb = None
    return (result_obj, (type(e), e, tb))

class ClearFrame(Exception): ...

cdef int clear_generator_frame(PyGenObject* gen):
    gen.gi_frame_state = get_frame_suspended()
    try:
        (<object> gen).throw(ClearFrame())
    except ClearFrame:
        pass
    except Exception as e:
        raise RuntimeError("error occur during frame clear") from e
    if gen.gi_frame_state == get_frame_cleared():
        return 0
    return -1
