## AI Summary

A file named pydevd_frame_tracing.py.


## Class: DummyTracingHolder

### Function: update_globals_dict(globals_dict)

### Function: _get_line_for_frame(frame)

### Function: _pydev_stop_at_break(line)

### Function: _pydev_needs_stop_at_break(line)

**Description:** We separate the functionality into 2 functions so that we can generate a bytecode which
generates a spurious line change so that we can do:

if _pydev_needs_stop_at_break():
    # Set line to line -1
    _pydev_stop_at_break()
    # then, proceed to go to the current line
    # (which will then trigger a line event).

### Function: set_trace_func(self, trace_func)
