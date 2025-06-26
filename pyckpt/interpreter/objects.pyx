import sys
from queue import SimpleQueue
from cpython.ref cimport PyObject, Py_INCREF

if (3, 11) <= sys.version_info <= (3, 12):
    from pyckpt.interpreter.objects cimport simplequeueobject_311 as simplequeueobject

cdef object _simple_queue_get_lst(PyObject* sq):
    cdef PyObject* lst = (<simplequeueobject*> sq).lst
    Py_INCREF(<object> lst)
    return <object> lst

def snapshot_simple_queue(sq: SimpleQueue):
    lst_pos = (<simplequeueobject*> sq).lst_pos
    return _simple_queue_get_lst(<PyObject*> sq)[lst_pos:]
