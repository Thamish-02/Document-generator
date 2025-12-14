## AI Summary

A file named pydevd_daemon_thread.py.


## Class: PyDBDaemonThread

### Function: _collect_load_names(func)

### Function: _patch_threading_to_hide_pydevd_threads()

**Description:** Patches the needed functions on the `threading` module so that the pydevd threads are hidden.

Note that we patch the functions __code__ to avoid issues if some code had already imported those
variables prior to the patching.

### Function: mark_as_pydevd_daemon_thread(thread)

### Function: run_as_pydevd_daemon_thread(py_db, func)

**Description:** Runs a function as a pydevd daemon thread (without any tracing in place).

### Function: __init__(self, py_db, target_and_args)

**Description:** :param target_and_args:
    tuple(func, args, kwargs) if this should be a function and args to run.
    -- Note: use through run_as_pydevd_daemon_thread().

### Function: py_db(self)

### Function: run(self)

### Function: _on_run(self)

### Function: do_kill_pydev_thread(self)

### Function: _stop_trace(self)

### Function: new_threading_enumerate()

### Function: pydevd_saved_threading_enumerate()

### Function: new_active_count()

### Function: new_threading_enumerate()

### Function: new_pick_some_non_daemon_thread()
