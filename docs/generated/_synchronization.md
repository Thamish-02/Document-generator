## AI Summary

A file named _synchronization.py.


### Function: current_async_library()

## Class: AsyncLock

**Description:** This is a standard lock.

In the sync case `Lock` provides thread locking.
In the async case `AsyncLock` provides async locking.

## Class: AsyncThreadLock

**Description:** This is a threading-only lock for no-I/O contexts.

In the sync case `ThreadLock` provides thread locking.
In the async case `AsyncThreadLock` is a no-op.

## Class: AsyncEvent

## Class: AsyncSemaphore

## Class: AsyncShieldCancellation

## Class: Lock

**Description:** This is a standard lock.

In the sync case `Lock` provides thread locking.
In the async case `AsyncLock` provides async locking.

## Class: ThreadLock

**Description:** This is a threading-only lock for no-I/O contexts.

In the sync case `ThreadLock` provides thread locking.
In the async case `AsyncThreadLock` is a no-op.

## Class: Event

## Class: Semaphore

## Class: ShieldCancellation

### Function: __init__(self)

### Function: setup(self)

**Description:** Detect if we're running under 'asyncio' or 'trio' and create
a lock with the correct implementation.

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self)

### Function: setup(self)

**Description:** Detect if we're running under 'asyncio' or 'trio' and create
a lock with the correct implementation.

### Function: set(self)

### Function: __init__(self, bound)

### Function: setup(self)

**Description:** Detect if we're running under 'asyncio' or 'trio' and create
a semaphore with the correct implementation.

### Function: __init__(self)

**Description:** Detect if we're running under 'asyncio' or 'trio' and create
a shielded scope with the correct implementation.

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self)

### Function: set(self)

### Function: wait(self, timeout)

### Function: __init__(self, bound)

### Function: acquire(self)

### Function: release(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)
