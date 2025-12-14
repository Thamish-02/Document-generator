## AI Summary

A file named pydevd_frame.py.


### Function: is_unhandled_exception(container_obj, py_db, frame, last_raise_line, raise_lines)

## Class: _TryExceptContainerObj

**Description:** A dumb container object just to contain the try..except info when needed. Meant to be
persistent among multiple PyDBFrames to the same code object.

## Class: PyDBFrame

**Description:** This makes the tracing for a given frame, so, the trace_dispatch
is used initially when we enter into a new context ('call') and then
is reused for the entire context.

### Function: should_stop_on_exception(py_db, info, frame, thread, arg, prev_user_uncaught_exc_info, is_unwind)

### Function: handle_exception(py_db, thread, frame, arg, exception_type)

### Function: set_suspend(self)

### Function: do_wait_suspend(self)

### Function: trace_exception(self, frame, event, arg)

### Function: handle_user_exception(self, frame)

### Function: get_func_name(self, frame)

### Function: _show_return_values(self, frame, arg)

### Function: _remove_return_values(self, py_db, frame)

### Function: _get_unfiltered_back_frame(self, py_db, frame)

### Function: _is_same_frame(self, target_frame, current_frame)

### Function: trace_dispatch(self, frame, event, arg)

### Function: get_smart_step_into_variant_from_frame_offset()

### Function: __init__(self)

### Function: __init__(self, args)
