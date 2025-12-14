## AI Summary

A file named _contextmanagers.py.


## Class: _SupportsCtxMgr

## Class: _SupportsAsyncCtxMgr

## Class: ContextManagerMixin

**Description:** Mixin class providing context manager functionality via a generator-based
implementation.

This class allows you to implement a context manager via :meth:`__contextmanager__`
which should return a generator. The mechanics are meant to mirror those of
:func:`@contextmanager <contextlib.contextmanager>`.

.. note:: Classes using this mix-in are not reentrant as context managers, meaning
    that once you enter it, you can't re-enter before first exiting it.

.. seealso:: :doc:`contextmanagers`

## Class: AsyncContextManagerMixin

**Description:** Mixin class providing async context manager functionality via a generator-based
implementation.

This class allows you to implement a context manager via
:meth:`__asynccontextmanager__`. The mechanics are meant to mirror those of
:func:`@asynccontextmanager <contextlib.asynccontextmanager>`.

.. note:: Classes using this mix-in are not reentrant as context managers, meaning
    that once you enter it, you can't re-enter before first exiting it.

.. seealso:: :doc:`contextmanagers`

### Function: __contextmanager__(self)

### Function: __asynccontextmanager__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: __contextmanager__(self)

**Description:** Implement your context manager logic here.

This method **must** be decorated with
:func:`@contextmanager <contextlib.contextmanager>`.

.. note:: Remember that the ``yield`` will raise any exception raised in the
    enclosed context block, so use a ``finally:`` block to clean up resources!

:return: a context manager object

### Function: __asynccontextmanager__(self)

**Description:** Implement your async context manager logic here.

This method **must** be decorated with
:func:`@asynccontextmanager <contextlib.asynccontextmanager>`.

.. note:: Remember that the ``yield`` will raise any exception raised in the
    enclosed context block, so use a ``finally:`` block to clean up resources!

:return: an async context manager object
