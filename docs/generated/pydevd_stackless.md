## AI Summary

A file named pydevd_stackless.py.


## Class: TaskletToLastId

**Description:** So, why not a WeakKeyDictionary?
The problem is that removals from the WeakKeyDictionary will create a new tasklet (as it adds a callback to
remove the key when it's garbage-collected), so, we can get into a recursion.

## Class: _TaskletInfo

### Function: get_tasklet_info(tasklet)

### Function: register_tasklet_info(tasklet)

### Function: _schedule_callback(prev, next)

**Description:** Called when a context is stopped or a new context is made runnable.

### Function: patch_stackless()

**Description:** This function should be called to patch the stackless module so that new tasklets are properly tracked in the
debugger.

### Function: __init__(self)

### Function: get(self, tasklet)

### Function: __setitem__(self, tasklet, last_id)

### Function: __init__(self, tasklet_weakref, tasklet)

### Function: update_name(self)

### Function: _schedule_callback(prev, next)

**Description:** Called when a context is stopped or a new context is made runnable.

### Function: setup(self)

**Description:** Called to run a new tasklet: rebind the creation so that we can trace it.

### Function: __call__(self)

**Description:** Called to run a new tasklet: rebind the creation so that we can trace it.

### Function: run()

### Function: set_schedule_callback(callable)

### Function: get_schedule_callback()

### Function: update_name(self)

### Function: new_f(old_f, args, kwargs)
