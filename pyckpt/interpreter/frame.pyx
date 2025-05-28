# cython: embedsignature=True, embedsignature.format=python

from types import CodeType, FrameType, FunctionType, TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject, Py_INCREF
from threading import Thread
from itertools import chain
from pyckpt.interpreter.cpython cimport *
from pyckpt.interpreter.cpython cimport (
    _Py_CODEUNIT,
    _PyErr_GetHandledException,
    _PyErr_SetHandledException,
    _PyEval_EvalFrameDefault,
)

import dis
import inspect
import logging
import sys

class NullObjectType:

    def __reduce__(self) -> str | tuple[Any, ...]:
        raise NotImplementedError("pickle NullObjectType is not allowed")

NullObject = NullObjectType()

cdef extern from "Python.h":
    int CO_GENERATOR
    int CO_COROUTINE
    int CO_ASYNC_GENERATOR
    int WAIT_LOCK

    ctypedef int (*Py_tracefunc)(PyObject *, PyFrameObject *, int, PyObject *)
    cdef int PyThread_acquire_lock(PyThread_type_lock lock, int wait_flag)
    cdef void PyThread_release_lock(PyThread_type_lock lock)
    cdef int _PyEval_SetTrace(PyThreadState *tstate, Py_tracefunc func, PyObject *arg)


if (3, 11) <= sys.version_info <= (3, 12):
    from pyckpt.interpreter.cpython cimport PyInterpreterFrame_311 as _PyInterpreterFrame
    from pyckpt.interpreter.cpython cimport GET_FRAME_311 as GET_FRAME
    from pyckpt.interpreter.cpython cimport PyFrame_InitializeSpecials_311 as _PyFrame_InitializeSpecials
    from pyckpt.interpreter.cpython cimport get_frame_owned_by_generator_311 as get_frame_owned_by_generator
    from pyckpt.interpreter.cpython cimport  _PyRuntimeState_311 as _PyRuntimeState
else:
    raise SystemError(f"unsupported python version: {sys.version_info}")

from pyckpt.interpreter.cpython cimport get_python_py_runtime

logger = logging.getLogger(__name__)


Analyzer = Callable[[FunctionType, int, bool], int]
ExceptionStates=Tuple[Type, Exception, TracebackType]

cdef extern from *:
    """
    #define OPCODE(x) (x >> 8)
    #define ARG(x) (x & 0xff)
    #include <stddef.h>
    #define container_of(ptr, type, member)                                        \
    ((type *)((char *)(ptr) - offsetof(type, member)))
    #define generator_of(ptr) (container_of(ptr, PyGenObject, gi_iframe))
    """

    cdef int OPCODE(int code_unit)
    cdef int ARG(int code_unit)
    cdef PyObject* generator_of(_PyInterpreterFrame* frame)

cdef PyObject* __PyErr_Occurred(PyThreadState* tstate):
    return tstate.curexc_type if tstate != NULL else NULL

def frame_specials_size():
    return (sizeof(_PyInterpreterFrame) - 1) // sizeof(PyObject*)


cdef PyThreadState* _get_thread_state_by_id(unsigned long thread_id):
    cdef PyThreadState *tstate = NULL;
    cdef _PyRuntimeState* runtime = <_PyRuntimeState*> get_python_py_runtime()
    if runtime == NULL:
        return NULL
    PyThread_acquire_lock(runtime.interpreters.mutex, WAIT_LOCK)
    tstate = PyInterpreterState_ThreadHead(PyThreadState_GET().interp)
    PyThread_release_lock(runtime.interpreters.mutex)
    while tstate != NULL:
        if tstate.thread_id != thread_id:
            tstate = PyThreadState_Next(tstate)
            continue
        return  tstate
    return tstate

cdef PyThreadState* get_thread_by_id(unsigned long thread_id):
    cdef PyThreadState* tstate = NULL
    if PyThread_get_thread_ident() == thread_id:
        return PyThreadState_GET()
    return _get_thread_state_by_id(thread_id)


cdef void init_locals_and_stack(
    void* frame_,
    list nlocals,
    list stack,
    int co_nlocalsplus,
):
    cdef _PyInterpreterFrame* frame = <_PyInterpreterFrame*> frame_
    # n local vars + 1 sentinel(NULL)
    for i in range(co_nlocalsplus + 1):
        frame.localsplus[i] = NULL
    assert len(nlocals) <= co_nlocalsplus
    for i, py_obj in enumerate(nlocals):
        if py_obj is not NullObject:
            Py_INCREF(py_obj)
            obj = <PyObject*>py_obj
            frame.localsplus[i] = obj
        else:
            frame.localsplus[i] = NULL
    cdef PyObject** stack_base = <PyObject**> &frame.localsplus[co_nlocalsplus]
    for i, py_obj in enumerate(stack):
        Py_INCREF(py_obj)
        obj = <PyObject*>py_obj
        stack_base[i] = obj
    frame.stacktop = co_nlocalsplus + len(stack)


def fetch_exception() -> ExceptionStates:
    cdef PyObject* p_type  = NULL
    cdef PyObject* p_value = NULL
    cdef PyObject* p_traceback = NULL

    PyErr_Fetch(&p_type, &p_value, &p_traceback)
    return (<object> p_type, <object> p_value, <object> p_traceback)

def restore_exception(states: ExceptionStates):
    PyErr_Restore(<PyObject*> states[0], <PyObject*> states[1], <PyObject*> states[2])


def eval_frame_at_lasti(
    func_obj: FunctionType,
    nlocals: List[Any],
    stack: List[Any],
    is_leaf: bool,
    ret_value: Any = None,
    prev_instr_offset = -1,
    exc_states: Optional[ExceptionStates] = None,
) -> Tuple[Any, Optional[ExceptionStates]]:
    cdef PyFunctionObject* func = <PyFunctionObject*>func_obj
    cdef PyThreadState* state = PyThreadState_GET()
    cdef PyCodeObject * code = <PyCodeObject*> func.func_code;
    cdef size_t size = code.co_nlocalsplus + code.co_stacksize + frame_specials_size();
    cdef _PyInterpreterFrame *frame = <_PyInterpreterFrame*> malloc(sizeof(PyObject*) * size)

    if not is_leaf:
        stack.append(ret_value)
    _PyFrame_InitializeSpecials(
        frame,
        <PyFunctionObject*> func,
        func.func_globals,
        code.co_nlocalsplus
    )
    frame.prev_instr = &(<_Py_CODEUNIT*>(code.co_code_adaptive))[prev_instr_offset]
    init_locals_and_stack(frame, nlocals, stack, code.co_nlocalsplus)
    do_exc = 0
    if exc_states is not None:
        if PyErr_Occurred() != NULL:
            _, exc_prev, _ = fetch_exception()
            PyErr_Clear()
            raise RuntimeError("eval a frame when exception has been raised")\
                from exc_prev
        restore_exception(exc_states)
        do_exc = 1
    cdef PyObject* result = _PyEval_EvalFrameDefault(state, frame, do_exc)
    if result == NULL:
        exc_states = fetch_exception()
        return NullObject, exc_states
    free(frame)
    return <object> result, None

def _offset_without_cache(
    code_array: List[dis.Instruction],
    offset_with_cache: int,
) -> int:
    CACHE = dis.opmap['CACHE']
    if offset_with_cache < 0:
        return offset_with_cache
    return sum(1 for c in code_array[:offset_with_cache] if c.opcode != CACHE)


# FIXME: these version-dependent functions should be placed separately.
CALL_INSTR_NAMES = ['CALL', 'CALL_FUNCTION_EX', 'YIELD_VALUE']
CALL_CODES = [dis.opmap[name] for name in CALL_INSTR_NAMES]
def is_call_instr(opcode: int):
    return opcode in CALL_CODES


def _fix_non_leaf_call(code_array: List[dis.Instruction], instr_offset):
    """
    Before CPython 3.11, python-python call is not "inlined"
    Hence we need to manually move instr_offset back to the 'CALL' instruction
    """
    CACHE = dis.opmap['CACHE']
    instr = code_array[instr_offset]
    if instr.opcode == CACHE:
        current = instr_offset - 1
        while code_array[current].opcode == CACHE:
            current -= 1
        instr_offset = current
    assert is_call_instr(code_array[instr_offset].opcode),\
        f"Invalid op {code_array[instr_offset]} at offset: {instr_offset}"
    return instr_offset

cdef int _check_generator(_PyInterpreterFrame* frame):
    cdef PyCodeObject* code = <PyCodeObject*> frame.f_func.func_code
    cdef int co_flags = code.co_flags
    cdef int mask = (CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR)
    cdef int flag = co_flags & mask
    if flag == 0:
        return 0
    if flag == inspect.CO_GENERATOR:
        assert frame.owner == get_frame_owned_by_generator()
        return 1
    else:
        return -1

def get_generator(frame: FrameType):
    cdef _PyInterpreterFrame* _frame = GET_FRAME(<PyFrameObject*> frame)
    if _check_generator(_frame) <= 0:
        return None
    gen = <object> generator_of(_frame)
    Py_INCREF(gen)
    return <object> gen

PyCapsule = Any
cdef object _snapshot_frame(void* frame_ptr, int is_leaf, object analyzer):
    cdef _PyInterpreterFrame* frame = <_PyInterpreterFrame*> frame_ptr
    if frame == NULL:
        raise RuntimeError("fail: PyCapsule_GetPointer")
    cdef PyFunctionObject* func = <PyFunctionObject*>(frame.f_func)
    cdef PyCodeObject* code = <PyCodeObject*> func.func_code
    cdef int nlocalsplus = code.co_nlocalsplus
    cdef _Py_CODEUNIT* code_start = <_Py_CODEUNIT*> code.co_code_adaptive

    instr_offset = frame.prev_instr - code_start
    assert instr_offset >= -1 and instr_offset < Py_SIZE(<PyObject*> code)
    code_array = list(dis.get_instructions(<object> code, show_caches=True))
    fixed_instr_offset = instr_offset
    if not is_leaf:
        fixed_instr_offset = _fix_non_leaf_call(
            code_array,
            instr_offset
        )
    is_generator = _check_generator(frame)
    stacksize = analyzer(
        <object> func,
        _offset_without_cache(code_array, fixed_instr_offset),
        is_generator > 0,
    )
    # Ignore the 'return value' of the ongoing 'CALL' instruction.
    if not is_leaf:
        stacksize -= 1
    generator: Optional[Generator] = None
    if is_generator > 0:
        generator = <object> generator_of(frame)
        Py_INCREF(generator)
    captured = {
        "func": <object> func,
        "nlocals": [
            <object> frame.localsplus[i]
                if frame.localsplus[i] != NULL else NullObject
            for i in range(nlocalsplus)
        ],
        "stack": [
            <object> frame.localsplus[nlocalsplus + i]
                if frame.localsplus[nlocalsplus + i] != NULL else NullObject
            for i in range(stacksize)
        ],
        "prev_instr_offset": instr_offset,
        "is_leaf": is_leaf,
    }
    for obj in chain(captured["nlocals"], captured["stack"]):
        Py_INCREF(obj)
    Py_INCREF(<object> func)
    Py_INCREF(generator)
    return captured

def snapshot(frame_obj: FrameType, is_leaf: bool, analyzer: Analyzer) -> Dict:
    cdef _PyInterpreterFrame* _frame = GET_FRAME(<PyFrameObject*>frame_obj)
    return _snapshot_frame(_frame, <int> is_leaf, analyzer)


def save_thread_state(thread: Thread):
    cdef PyThreadState* tstate
    cdef PyObject* exc
    tstate = get_thread_by_id(<unsigned long> thread.ident)
    if tstate == NULL:
        raise RuntimeError("fail to fetch python runtime information")
    if __PyErr_Occurred(tstate) != NULL:
        raise RuntimeError("capture a thread while exception is happening")
    exc = _PyErr_GetHandledException(tstate)
    exc_obj = <object> exc if exc != NULL else None
    return {
        "exception": exc_obj
    }

def restore_thread_state(state: Dict):
    exc = state["exception"]
    if exc is None:
        _PyErr_SetHandledException(PyThreadState_GET(), NULL)
        return
    if PyErr_Occurred() != NULL or PyErr_GetHandledException() != NULL:
        raise RuntimeError("set thread state when there is exception pending")
    _PyErr_SetHandledException(PyThreadState_GET(), <PyObject *> exc)
    state["exception"] = None


cdef int trace_trampoline(PyObject *callback, PyFrameObject *frame, int what, PyObject *arg) noexcept:
    (<object> callback)(<object> frame, what, <object> arg)
    return 0


def set_trace_all_threads(func: Optional[FunctionType]):
    cdef _PyRuntimeState* runtime = <_PyRuntimeState*> get_python_py_runtime()
    cdef PyThreadState* tstate
    if runtime == NULL:
        raise RuntimeError("filed to fetch python runtime")

    PyThread_acquire_lock(runtime.interpreters.mutex, WAIT_LOCK)
    tstate = PyInterpreterState_ThreadHead(PyThreadState_GET().interp)
    PyThread_release_lock(runtime.interpreters.mutex)

    while tstate != NULL:
        _PyEval_SetTrace(tstate, trace_trampoline, <PyObject*> func)
        tstate = PyThreadState_Next(tstate)
