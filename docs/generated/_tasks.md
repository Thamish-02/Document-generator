## AI Summary

A file named _tasks.py.


## Class: _IgnoredTaskStatus

## Class: CancelScope

**Description:** Wraps a unit of work that can be made separately cancellable.

:param deadline: The time (clock value) when this scope is cancelled automatically
:param shield: ``True`` to shield the cancel scope from external cancellation

### Function: fail_after(delay, shield)

**Description:** Create a context manager which raises a :class:`TimeoutError` if does not finish in
time.

:param delay: maximum allowed time (in seconds) before raising the exception, or
    ``None`` to disable the timeout
:param shield: ``True`` to shield the cancel scope from external cancellation
:return: a context manager that yields a cancel scope
:rtype: :class:`~typing.ContextManager`\[:class:`~anyio.CancelScope`\]

### Function: move_on_after(delay, shield)

**Description:** Create a cancel scope with a deadline that expires after the given delay.

:param delay: maximum allowed time (in seconds) before exiting the context block, or
    ``None`` to disable the timeout
:param shield: ``True`` to shield the cancel scope from external cancellation
:return: a cancel scope

### Function: current_effective_deadline()

**Description:** Return the nearest deadline among all the cancel scopes effective for the current
task.

:return: a clock value from the event loop's internal clock (or ``float('inf')`` if
    there is no deadline in effect, or ``float('-inf')`` if the current scope has
    been cancelled)
:rtype: float

### Function: create_task_group()

**Description:** Create a task group.

:return: a task group

### Function: started(self, value)

### Function: __new__(cls)

### Function: cancel(self, reason)

**Description:** Cancel this scope immediately.

:param reason: a message describing the reason for the cancellation

### Function: deadline(self)

**Description:** The time (clock value) when this scope is cancelled automatically.

Will be ``float('inf')`` if no timeout has been set.

### Function: deadline(self, value)

### Function: cancel_called(self)

**Description:** ``True`` if :meth:`cancel` has been called.

### Function: cancelled_caught(self)

**Description:** ``True`` if this scope suppressed a cancellation exception it itself raised.

This is typically used to check if any work was interrupted, or to see if the
scope was cancelled due to its deadline being reached. The value will, however,
only be ``True`` if the cancellation was triggered by the scope itself (and not
an outer scope).

### Function: shield(self)

**Description:** ``True`` if this scope is shielded from external cancellation.

While a scope is shielded, it will not receive cancellations from outside.

### Function: shield(self, value)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)
