## AI Summary

A file named pydevd_constants.py.


## Class: DebugInfoHolder

### Function: version_str(v)

### Function: is_true_in_env(env_key)

### Function: as_float_in_env(env_key, default)

### Function: as_int_in_env(env_key, default)

### Function: protect_libraries_from_patching()

**Description:** In this function we delete some modules from `sys.modules` dictionary and import them again inside
  `_pydev_saved_modules` in order to save their original copies there. After that we can use these
  saved modules within the debugger to protect them from patching by external libraries (e.g. gevent).

### Function: after_fork()

**Description:** Must be called after a fork operation (will reset the ForkSafeLock).

### Function: as_str(s)

### Function: filter_all_warnings()

### Function: silence_warnings_decorator(func)

### Function: sorted_dict_repr(d)

### Function: iter_chars(b)

### Function: get_pid()

### Function: clear_cached_thread_id(thread)

### Function: _get_or_compute_thread_id_with_lock(thread, is_current_thread)

### Function: get_current_thread_id(thread)

**Description:** Note: the difference from get_current_thread_id to get_thread_id is that
for the current thread we can get the thread id while the thread.ident
is still not set in the Thread instance.

### Function: get_thread_id(thread)

### Function: set_thread_id(thread, thread_id)

## Class: Null

**Description:** Gotten from: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/68205

## Class: KeyifyList

### Function: call_only_once(func)

**Description:** To be used as a decorator

@call_only_once
def func():
    print 'Calling func only this time'

Actually, in PyDev it must be called as:

func = call_only_once(func) to support older versions of Python.

## Class: _GlobalSettings

### Function: set_protocol(protocol)

### Function: get_protocol()

### Function: is_json_protocol()

## Class: GlobalDebuggerHolder

**Description:** Holder for the global debugger.

### Function: get_global_debugger()

### Function: set_global_debugger(dbg)

### Function: ForkSafeLock(rlock)

## Class: ForkSafeLock

**Description:** A lock which is fork-safe (when a fork is done, `pydevd_constants.after_fork()`
should be called to reset the locks in the new process to avoid deadlocks
from a lock which was locked during the fork).

Note:
    Unlike `threading.Lock` this class is not completely atomic, so, doing:

    lock = ForkSafeLock()
    with lock:
        ...

    is different than using `threading.Lock` directly because the tracing may
    find an additional function call on `__enter__` and on `__exit__`, so, it's
    not recommended to use this in all places, only where the forking may be important
    (so, for instance, the locks on PyDB should not be changed to this lock because
    of that -- and those should all be collected in the new process because PyDB itself
    should be completely cleared anyways).

    It's possible to overcome this limitation by using `ForkSafeLock.acquire` and
    `ForkSafeLock.release` instead of the context manager (as acquire/release are
    bound to the original implementation, whereas __enter__/__exit__ is not due to Python
    limitations).

### Function: new_func()

### Function: NO_FTRACE(frame, event, arg)

### Function: _temp_trace(frame, event, arg)

### Function: _check_ftrace_set_none()

**Description:** Will throw an error when executing a line event

### Function: __init__(self)

### Function: __call__(self)

### Function: __enter__(self)

### Function: __exit__(self)

### Function: __getattr__(self, mname)

### Function: __setattr__(self, name, value)

### Function: __delattr__(self, name)

### Function: __repr__(self)

### Function: __str__(self)

### Function: __len__(self)

### Function: __getitem__(self)

### Function: __setitem__(self)

### Function: write(self)

### Function: __nonzero__(self)

### Function: __iter__(self)

### Function: __init__(self, inner, key)

### Function: __len__(self)

### Function: __getitem__(self, k)

### Function: new_func()

### Function: get_frame()

### Function: get_frame()

### Function: _current_frames()

### Function: __init__(self, rlock)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: _init(self)

### Function: NO_FTRACE(frame, event, arg)

### Function: _current_frames()

### Function: NO_FTRACE(frame, event, arg)
