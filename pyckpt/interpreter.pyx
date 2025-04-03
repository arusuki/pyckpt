import sys
from types import CodeType, FrameType, FunctionType
from typing import Any, Dict, List 
from libc.stdlib cimport malloc, free
from pyckpt.analyzer import Analyzer

import pyckpt.frame
import dis
import bytecode


def frame_specials_size():
    return (sizeof(_PyInterpreterFrame) - 1) // sizeof(PyObject*)

cdef extern from "defs.h":

    cdef int OPCODE(int code_unit)

    cdef int ARG(int code_unit)

cdef extern from "internal/pycore_pystate.h":

    cdef PyThreadState * _PyThreadState_GET()


cdef extern from "internal/pycore_frame.h":

    ctypedef void PyObject

    cdef int Py_SIZE(PyObject*)

    cdef void Py_INCREF(PyObject*)

    ctypedef int _Py_CODEUNIT

    ctypedef struct PyCodeObject:
        int co_stacksize
        int co_nlocalsplus
        char[1] co_code_adaptive


    ctypedef struct PyFunctionObject:
        PyCodeObject *func_code
        PyObject *func_globals


    ctypedef struct PyThreadState:
        pass

    ctypedef struct _PyInterpreterFrame:
        PyFunctionObject *f_func
        _Py_CODEUNIT *prev_instr;
        int stacktop;
        PyObject* localsplus[1];

    ctypedef struct PyFrameObject:
        _PyInterpreterFrame *f_frame
    
    cdef void _PyFrame_InitializeSpecials(_PyInterpreterFrame *frame, PyFunctionObject *func, PyObject *locals, int nlocalsplus)



cdef extern PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, _PyInterpreterFrame *frame, int throwflag)

cdef _init_locals_and_stack(
    _PyInterpreterFrame* frame,
    nlocals: List[Any],
    stack: List[Any],
    co_nlocalsplus: int, 
    co_stacksize: int,
):
    # n local vars + 1 sentinel(NULL)
    for i in range(co_nlocalsplus + 1):
        frame.localsplus[i] = NULL

    assert len(nlocals) <= co_nlocalsplus

    for i, pyobj in enumerate(nlocals):
        if pyobj is not pyckpt.frame.NullObject:
            obj = <PyObject*>pyobj
            frame.localsplus[i] = obj
        else:
            frame.localsplus[i] = NULL
    
    cdef PyObject** stack_base = <PyObject**> &frame.localsplus[co_nlocalsplus]
    
    for i, pyobj in enumerate(stack):
        obj = <PyObject*>pyobj
        stack_base[i] = obj

    frame.stacktop = co_nlocalsplus + len(stack) 



def eval_frame_at_lasti(
    func_obj: FunctionType,
    nlocals: List[Any],
    stack: List[Any],
    prev_instr_offset,
    is_leaf: bool,
    ret_value: Any,
):
    cdef PyFunctionObject* func = <PyFunctionObject*>func_obj
    cdef PyThreadState* state = _PyThreadState_GET()
    cdef PyCodeObject * code = func.func_code;
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

    _init_locals_and_stack(frame, nlocals, stack, code.co_nlocalsplus, code.co_stacksize)
    
    result = <object> _PyEval_EvalFrameDefault(state, frame, 0)

    free(frame)

    return result

def _offset_without_cache(
    code_array: List[dis.Instruction],
    offset_with_cache: int,
) -> int:
    CACHE = dis.opmap['CACHE']

    if offset_with_cache < 0:
        return offset_with_cache
    
    return sum(1 for c in code_array[:offset_with_cache] if c.opcode != CACHE)


def _fix_non_leaf_call(code_array: List[dis.Instruction], instr_offset):
    CALL = dis.opmap['CALL']
    CACHE = dis.opmap['CACHE']
    instr = code_array[instr_offset]

    if instr.opcode == CACHE:
        current = instr_offset - 1
        while code_array[current].opcode == CACHE:
            current -= 1
        assert code_array[current].opcode == CALL, f"Invalid op {code_array[current]} at offset: {current} "
    else:
        # Since CPython 3.11, python-python call is "inlined"
        # Manually adjust offset to move across the cache instructions.
        assert instr.opcode == CALL
        instr_offset += 1
        while code_array[instr_offset].opcode == CACHE:
            instr_offset += 1
        instr_offset -= 1
    
    return instr_offset


def snapshot(frame_obj: FrameType, is_leaf: bool, analyzer: Analyzer) -> Dict:
    cdef _PyInterpreterFrame* frame = (<PyFrameObject*>frame_obj).f_frame
    cdef PyFunctionObject* func = <PyFunctionObject*>(frame.f_func)
    cdef PyCodeObject* code = <PyCodeObject*> func.func_code
    cdef int nlocalsplus = code.co_nlocalsplus
    cdef _Py_CODEUNIT* code_start = <_Py_CODEUNIT*> code.co_code_adaptive

    instr_offset = frame.prev_instr - code_start
    assert instr_offset >= -1 and instr_offset < Py_SIZE(code)

    code_array = list(dis.get_instructions(<object> code, show_caches=True))

    if not is_leaf:
        instr_offset = _fix_non_leaf_call(
            code_array,
            instr_offset
        )

    stacksize = analyzer(<object> func, _offset_without_cache(code_array, instr_offset))

    if not is_leaf:
        # Ignore the 'return value' of the ongoing 'CALL' instruction.
        stacksize -= 1 

    return {
        "func": <object> func,
        "nlocals": [ 
            <object> frame.localsplus[i]
                if frame.localsplus[i] != NULL else pyckpt.frame.NullObject
            for i in range(nlocalsplus)
        ],
        "stack": [
            <object> frame.localsplus[nlocalsplus + i]
                if frame.localsplus[nlocalsplus + i] != NULL else pyckpt.frame.NullObject 
            for i in range(stacksize)
        ],
        "prev_instr_offset": instr_offset,
        "is_leaf": is_leaf,
    }
