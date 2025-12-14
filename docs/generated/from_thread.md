## AI Summary

A file named from_thread.py.


### Function: _token_or_error(token)

### Function: run(func)

**Description:** Call a coroutine function from a worker thread.

:param func: a coroutine function
:param args: positional arguments for the callable
:param token: an event loop token to use to get back to the event loop thread
    (required if calling this function from outside an AnyIO worker thread)
:return: the return value of the coroutine function
:raises MissingTokenError: if no token was provided and called from outside an
    AnyIO worker thread
:raises RunFinishedError: if the event loop tied to ``token`` is no longer running

.. versionchanged:: 4.11.0
    Added the ``token`` parameter.

### Function: run_sync(func)

**Description:** Call a function in the event loop thread from a worker thread.

:param func: a callable
:param args: positional arguments for the callable
:param token: an event loop token to use to get back to the event loop thread
    (required if calling this function from outside an AnyIO worker thread)
:return: the return value of the callable
:raises MissingTokenError: if no token was provided and called from outside an
    AnyIO worker thread
:raises RunFinishedError: if the event loop tied to ``token`` is no longer running

.. versionchanged:: 4.11.0
    Added the ``token`` parameter.

## Class: _BlockingAsyncContextManager

## Class: _BlockingPortalTaskStatus

## Class: BlockingPortal

**Description:** An object that lets external threads run code in an asynchronous event loop.

## Class: BlockingPortalProvider

**Description:** A manager for a blocking portal. Used as a context manager. The first thread to
enter this context manager causes a blocking portal to be started with the specific
parameters, and the last thread to exit causes the portal to be shut down. Thus,
there will be exactly one blocking portal running in this context as long as at
least one thread has entered this context manager.

The parameters are the same as for :func:`~anyio.run`.

:param backend: name of the backend
:param backend_options: backend options

.. versionadded:: 4.4

### Function: start_blocking_portal(backend, backend_options)

**Description:** Start a new event loop in a new thread and run a blocking portal in its main task.

The parameters are the same as for :func:`~anyio.run`.

:param backend: name of the backend
:param backend_options: backend options
:param name: name of the thread
:return: a context manager that yields a blocking portal

.. versionchanged:: 3.0
    Usage as a context manager is now required.

### Function: check_cancelled()

**Description:** Check if the cancel scope of the host task's running the current worker thread has
been cancelled.

If the host task's current cancel scope has indeed been cancelled, the
backend-specific cancellation exception will be raised.

:raises RuntimeError: if the current thread was not spawned by
    :func:`.to_thread.run_sync`

### Function: __init__(self, async_cm, portal)

### Function: __enter__(self)

### Function: __exit__(self, __exc_type, __exc_value, __traceback)

### Function: __init__(self, future)

### Function: started(self, value)

### Function: __new__(cls)

### Function: __init__(self)

### Function: _check_running(self)

### Function: _spawn_task_from_thread(self, func, args, kwargs, name, future)

**Description:** Spawn a new task using the given callable.

Implementers must ensure that the future is resolved when the task finishes.

:param func: a callable
:param args: positional arguments to be passed to the callable
:param kwargs: keyword arguments to be passed to the callable
:param name: name of the task (will be coerced to a string if not ``None``)
:param future: a future that will resolve to the return value of the callable,
    or the exception raised during its execution

### Function: call(self, func)

### Function: call(self, func)

### Function: call(self, func)

**Description:** Call the given function in the event loop thread.

If the callable returns a coroutine object, it is awaited on.

:param func: any callable
:raises RuntimeError: if the portal is not running or if this method is called
    from within the event loop thread

### Function: start_task_soon(self, func)

### Function: start_task_soon(self, func)

### Function: start_task_soon(self, func)

**Description:** Start a task in the portal's task group.

The task will be run inside a cancel scope which can be cancelled by cancelling
the returned future.

:param func: the target function
:param args: positional arguments passed to ``func``
:param name: name of the task (will be coerced to a string if not ``None``)
:return: a future that resolves with the return value of the callable if the
    task completes successfully, or with the exception raised in the task
:raises RuntimeError: if the portal is not running or if this method is called
    from within the event loop thread
:rtype: concurrent.futures.Future[T_Retval]

.. versionadded:: 3.0

### Function: start_task(self, func)

**Description:** Start a task in the portal's task group and wait until it signals for readiness.

This method works the same way as :meth:`.abc.TaskGroup.start`.

:param func: the target function
:param args: positional arguments passed to ``func``
:param name: name of the task (will be coerced to a string if not ``None``)
:return: a tuple of (future, task_status_value) where the ``task_status_value``
    is the value passed to ``task_status.started()`` from within the target
    function
:rtype: tuple[concurrent.futures.Future[T_Retval], Any]

.. versionadded:: 3.0

### Function: wrap_async_context_manager(self, cm)

**Description:** Wrap an async context manager as a synchronous context manager via this portal.

Spawns a task that will call both ``__aenter__()`` and ``__aexit__()``, stopping
in the middle until the synchronous context manager exits.

:param cm: an asynchronous context manager
:return: a synchronous context manager

.. versionadded:: 2.1

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: run_blocking_portal()

### Function: callback(f)

### Function: task_done(future)
