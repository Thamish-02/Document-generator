## AI Summary

A file named pydevd_trace_dispatch_regular.py.


### Function: notify_skipped_step_in_because_of_filters(py_db, frame)

### Function: fix_top_level_trace_and_get_trace_func(py_db, frame)

### Function: trace_dispatch(py_db, frame, event, arg)

## Class: TopLevelThreadTracerOnlyUnhandledExceptions

## Class: TopLevelThreadTracerNoBackFrame

**Description:** This tracer is pretty special in that it's dealing with a frame without f_back (i.e.: top frame
on remote attach or QThread).

This means that we have to carefully inspect exceptions to discover whether the exception will
be unhandled or not (if we're dealing with an unhandled exception we need to stop as unhandled,
otherwise we need to use the regular tracer -- unfortunately the debugger has little info to
work with in the tracing -- see: https://bugs.python.org/issue34099, so, we inspect bytecode to
determine if some exception will be traced or not... note that if this is not available -- such
as on Jython -- we consider any top-level exception to be unnhandled).

## Class: ThreadTracer

### Function: __init__(self, args)

### Function: trace_unhandled_exceptions(self, frame, event, arg)

### Function: get_trace_dispatch_func(self)

### Function: __init__(self, frame_trace_dispatch, args)

### Function: trace_dispatch_and_unhandled_exceptions(self, frame, event, arg)

### Function: get_trace_dispatch_func(self)

### Function: __init__(self, args)

### Function: __call__(self, frame, event, arg)

**Description:** This is the callback used when we enter some context in the debugger.

We also decorate the thread we are in with info about the debugging.
The attributes added are:
    pydev_state
    pydev_step_stop
    pydev_step_cmd
    pydev_notify_kill

:param PyDB py_db:
    This is the global debugger (this method should actually be added as a method to it).

### Function: __call__(self, frame, event, arg)

### Function: fix_top_level_trace_and_get_trace_func()
