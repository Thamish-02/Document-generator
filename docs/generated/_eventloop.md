## AI Summary

A file named _eventloop.py.


### Function: run(func)

**Description:** Run the given coroutine function in an asynchronous event loop.

The current thread must not be already running an event loop.

:param func: a coroutine function
:param args: positional arguments to ``func``
:param backend: name of the asynchronous event loop implementation – currently
    either ``asyncio`` or ``trio``
:param backend_options: keyword arguments to call the backend ``run()``
    implementation with (documented :ref:`here <backend options>`)
:return: the return value of the coroutine function
:raises RuntimeError: if an asynchronous event loop is already running in this
    thread
:raises LookupError: if the named backend is not found

### Function: current_time()

**Description:** Return the current value of the event loop's internal clock.

:return: the clock value (seconds)

### Function: get_all_backends()

**Description:** Return a tuple of the names of all built-in backends.

### Function: get_cancelled_exc_class()

**Description:** Return the current async library's cancellation exception class.

### Function: claim_worker_thread(backend_class, token)

### Function: get_async_backend(asynclib_name)
