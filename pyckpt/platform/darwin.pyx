from threading import Thread

cdef extern from "defs.h":

    ctypedef int mach_port_t
    ctypedef int kern_return_t
    ctypedef int mach_port_kernel_object_t
    ctypedef int ipc_space_read_t

    cdef int KERN_SUCCESS

    cdef kern_return_t mach_port_kernel_object (ipc_space_read_t, mach_port_t, unsigned *, unsigned *)
    cdef mach_port_t   mach_task_self          ()

    cdef kern_return_t thread_suspend          (mach_port_t)
    cdef kern_return_t thread_resume           (mach_port_t)

cdef extern from "<pthread.h>":

    ctypedef void* pthread_t

    cdef mach_port_t pthread_mach_thread_np(pthread_t)


IKOT_THREAD_CONTROL=0x1


cdef int is_valid_thread_port(ident: pthread_t):
    cdef kern_return_t kr
    cdef unsigned object_type = 0
    cdef unsigned object_addr = 0
    cdef mach_port_t port = pthread_mach_thread_np(ident)

    kr = mach_port_kernel_object(mach_task_self(), port, &object_type, &object_addr)

    return -1 \
    if   kr != KERN_SUCCESS or object_type != IKOT_THREAD_CONTROL \
    else 0

cdef int _suspend_thread(ident: pthread_t):
    cdef mach_port_t port = pthread_mach_thread_np(ident)
    kr = thread_suspend(port)
    return -1 if kr != KERN_SUCCESS else 0

cdef int _resume_thread(ident: pthread_t):
    cdef mach_port_t port = pthread_mach_thread_np(ident)
    kr = thread_resume(port)
    return -1 if kr != KERN_SUCCESS else 0


def is_valid_running_thread(t: Thread) -> bool:
    if not t.is_alive():
        return False

    cdef unsigned long ident = t.ident
    # # Here assumes the typeof 'ident' filed is 'pthread_t', which is 
    # # undocumented in Python's docs.
    return is_valid_thread_port(<pthread_t> ident) == 0


def suspend_thread(t: Thread):
    assert t.is_alive(), "suspend not-started/finished thread"

    cdef unsigned long ident = t.ident

    return _suspend_thread(<pthread_t> ident) == 0

def resume_thread(t: Thread):
    assert t.is_alive(), "resume not-started/finished thread"

    cdef unsigned long ident = t.ident

    return _resume_thread(<pthread_t> ident) == 0

