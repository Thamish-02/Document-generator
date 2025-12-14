## AI Summary

A file named singleton.py.


## Class: Singleton

**Description:** A base class for a class of a singleton object.

For any derived class T, the first invocation of T() will create the instance,
and any future invocations of T() will return that instance.

Concurrent invocations of T() from different threads are safe.

## Class: ThreadSafeSingleton

**Description:** A singleton that incorporates a lock for thread-safe access to its members.

The lock can be acquired using the context manager protocol, and thus idiomatic
use is in conjunction with a with-statement. For example, given derived class T::

    with T() as t:
        t.x = t.frob(t.y)

All access to the singleton from the outside should follow this pattern for both
attributes and method calls. Singleton members can assume that self is locked by
the caller while they're executing, but recursive locking of the same singleton
on the same thread is also permitted.

### Function: threadsafe_method(func)

**Description:** Marks a method of a ThreadSafeSingleton-derived class as inherently thread-safe.

A method so marked must either not use any singleton state, or lock it appropriately.

### Function: autolocked_method(func)

**Description:** Automatically synchronizes all calls of a method of a ThreadSafeSingleton-derived
class by locking the singleton for the duration of each call.

### Function: __new__(cls)

### Function: __init__(self)

**Description:** Initializes the singleton instance. Guaranteed to only be invoked once for
any given type derived from Singleton.

If shared=False, the caller is requesting a singleton instance for their own
exclusive use. This is only allowed if the singleton has not been created yet;
if so, it is created and marked as being in exclusive use. While it is marked
as such, all attempts to obtain an existing instance of it immediately raise
an exception. The singleton can eventually be promoted to shared use by calling
share() on it.

### Function: __enter__(self)

**Description:** Lock this singleton to prevent concurrent access.

### Function: __exit__(self, exc_type, exc_value, exc_tb)

**Description:** Unlock this singleton to allow concurrent access.

### Function: share(self)

**Description:** Share this singleton, if it was originally created with shared=False.

### Function: __init__(self)

### Function: assert_locked(self)

### Function: __getattribute__(self, name)

### Function: __setattr__(self, name, value)

### Function: lock_and_call(self)
