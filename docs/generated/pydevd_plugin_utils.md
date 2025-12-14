## AI Summary

A file named pydevd_plugin_utils.py.


### Function: load_plugins()

### Function: bind_func_to_method(func, obj, method_name)

## Class: PluginManager

### Function: __init__(self, main_debugger)

### Function: add_breakpoint(self, func_name)

### Function: activate(self, plugin)

### Function: after_breakpoints_consolidated(self, py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)

### Function: remove_exception_breakpoint(self, py_db, exception_type, exception)

**Description:** :param exception_type: 'django', 'jinja2' (can be extended)

### Function: remove_all_exception_breakpoints(self, py_db)

### Function: get_breakpoints(self, py_db, breakpoint_type)

**Description:** :param breakpoint_type: 'django-line', 'jinja2-line'

### Function: can_skip(self, py_db, frame)

### Function: required_events_breakpoint(self)

### Function: required_events_stepping(self)

### Function: is_tracked_frame(self, frame)

### Function: has_exception_breaks(self, py_db)

### Function: has_line_breaks(self, py_db)

### Function: cmd_step_into(self, py_db, frame, event, info, thread, stop_info, stop)

**Description:** :param stop_info: in/out information. If it should stop then it'll be
    filled by the plugin.
:param stop: whether the stop has already been flagged for this frame.
:returns:
    tuple(stop, plugin_stop)

### Function: cmd_step_over(self, py_db, frame, event, info, thread, stop_info, stop)

### Function: stop(self, py_db, frame, event, thread, stop_info, arg, step_cmd)

**Description:** The way this works is that the `cmd_step_into` or `cmd_step_over`
is called which then fills the `stop_info` and then this method
is called to do the actual stop.

### Function: get_breakpoint(self, py_db, frame, event, info)

### Function: suspend(self, py_db, thread, frame, bp_type)

**Description:** :param bp_type: 'django' or 'jinja2'

:return:
    The frame for the suspend or None if it should not be suspended.

### Function: exception_break(self, py_db, frame, thread, arg, is_unwind)

### Function: change_variable(self, frame, attr, expression, scope)
