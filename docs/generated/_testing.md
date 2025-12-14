## AI Summary

A file named _testing.py.


## Class: TaskInfo

**Description:** Represents an asynchronous task.

:ivar int id: the unique identifier of the task
:ivar parent_id: the identifier of the parent task, if any
:vartype parent_id: Optional[int]
:ivar str name: the description of the task (if any)
:ivar ~collections.abc.Coroutine coro: the coroutine object of the task

### Function: get_current_task()

**Description:** Return the current task.

:return: a representation of the current task

### Function: get_running_tasks()

**Description:** Return a list of running tasks in the current event loop.

:return: a list of task info objects

### Function: __init__(self, id, parent_id, name, coro)

### Function: __eq__(self, other)

### Function: __hash__(self)

### Function: __repr__(self)

### Function: has_pending_cancellation(self)

**Description:** Return ``True`` if the task has a cancellation pending, ``False`` otherwise.
