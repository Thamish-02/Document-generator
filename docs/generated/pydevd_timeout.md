## AI Summary

A file named pydevd_timeout.py.


## Class: _TimeoutThread

**Description:** The idea in this class is that it should be usually stopped waiting
for the next event to be called (paused in a threading.Event.wait).

When a new handle is added it sets the event so that it processes the handles and
then keeps on waiting as needed again.

This is done so that it's a bit more optimized than creating many Timer threads.

## Class: _OnTimeoutHandle

## Class: TimeoutTracker

**Description:** This is a helper class to track the timeout of something.

### Function: create_interrupt_this_thread_callback()

**Description:** The idea here is returning a callback that when called will generate a KeyboardInterrupt
in the thread that called this function.

If this is the main thread, this means that it'll emulate a Ctrl+C (which may stop I/O
and sleep operations).

For other threads, this will call PyThreadState_SetAsyncExc to raise
a KeyboardInterrupt before the next instruction (so, it won't really interrupt I/O or
sleep operations).

:return callable:
    Returns a callback that will interrupt the current thread (this may be called
    from an auxiliary thread).

### Function: __init__(self, py_db)

### Function: _on_run(self)

### Function: process_handles(self)

**Description:** :return int:
    Returns the time we should be waiting for to process the next event properly.

### Function: do_kill_pydev_thread(self)

### Function: add_on_timeout_handle(self, handle)

### Function: __init__(self, tracker, abs_timeout, on_timeout, kwargs)

### Function: exec_on_timeout(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: __str__(self)

### Function: __init__(self, py_db)

### Function: call_on_timeout(self, timeout, on_timeout, kwargs)

**Description:** This can be called regularly to always execute the given function after a given timeout:

call_on_timeout(py_db, 10, on_timeout)


Or as a context manager to stop the method from being called if it finishes before the timeout
elapses:

with call_on_timeout(py_db, 10, on_timeout):
    ...

Note: the callback will be called from a PyDBDaemonThread.

### Function: raise_on_this_thread()

### Function: raise_on_this_thread()
