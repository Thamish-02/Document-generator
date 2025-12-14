## AI Summary

A file named events.py.


## Class: EventManager

**Description:** Manage a collection of events and a sequence of callbacks for each.

This is attached to :class:`~IPython.core.interactiveshell.InteractiveShell`
instances as an ``events`` attribute.

.. note::

   This API is experimental in IPython 2.0, and may be revised in future versions.

### Function: _define_event(callback_function)

### Function: pre_execute()

**Description:** Fires before code is executed in response to user/frontend action.

This includes comm and widget messages and silent execution, as well as user
code cells.

### Function: pre_run_cell(info)

**Description:** Fires before user-entered code runs.

Parameters
----------
info : :class:`~IPython.core.interactiveshell.ExecutionInfo`
    An object containing information used for the code execution.

### Function: post_execute()

**Description:** Fires after code is executed in response to user/frontend action.

This includes comm and widget messages and silent execution, as well as user
code cells.

### Function: post_run_cell(result)

**Description:** Fires after user-entered code runs.

Parameters
----------
result : :class:`~IPython.core.interactiveshell.ExecutionResult`
    The object which will be returned as the execution result.

### Function: shell_initialized(ip)

**Description:** Fires after initialisation of :class:`~IPython.core.interactiveshell.InteractiveShell`.

This is before extensions and startup scripts are loaded, so it can only be
set by subclassing.

Parameters
----------
ip : :class:`~IPython.core.interactiveshell.InteractiveShell`
    The newly initialised shell.

### Function: __init__(self, shell, available_events, print_on_error)

**Description:** Initialise the :class:`CallbackManager`.

Parameters
----------
shell
    The :class:`~IPython.core.interactiveshell.InteractiveShell` instance
available_events
    An iterable of names for callback events.
print_on_error:
    A boolean flag to set whether the EventManager will print a warning which a event errors.

### Function: register(self, event, function)

**Description:** Register a new event callback.

Parameters
----------
event : str
    The event for which to register this callback.
function : callable
    A function to be called on the given event. It should take the same
    parameters as the appropriate callback prototype.

Raises
------
TypeError
    If ``function`` is not callable.
KeyError
    If ``event`` is not one of the known events.

### Function: unregister(self, event, function)

**Description:** Remove a callback from the given event.

### Function: trigger(self, event)

**Description:** Call callbacks for ``event``.

Any additional arguments are passed to all callbacks registered for this
event. Exceptions raised by callbacks are caught, and a message printed.
