## AI Summary

A file named pydevd_suspended_frames.py.


## Class: _AbstractVariable

## Class: _ObjectVariable

### Function: sorted_variables_key(obj)

## Class: _FrameVariable

## Class: _FramesTracker

**Description:** This is a helper class to be used to track frames when a thread becomes suspended.

## Class: SuspendedFramesManager

### Function: __init__(self, py_db)

### Function: get_name(self)

### Function: get_value(self)

### Function: get_variable_reference(self)

### Function: get_var_data(self, fmt, context)

**Description:** :param dict fmt:
    Format expected by the DAP (keys: 'hex': bool, 'rawString': bool)

:param context:
    This is the context in which the variable is being requested. Valid values:
        "watch",
        "repl",
        "hover",
        "clipboard"

### Function: get_children_variables(self, fmt, scope)

### Function: get_child_variable_named(self, name, fmt, scope)

### Function: _group_entries(self, lst, handle_return_values)

### Function: __init__(self, py_db, name, value, register_variable, is_return_value, evaluate_name, frame)

### Function: get_children_variables(self, fmt, scope)

### Function: change_variable(self, name, value, py_db, fmt, scope)

### Function: __init__(self, py_db, frame, register_variable)

### Function: change_variable(self, name, value, py_db, fmt, scope)

### Function: get_children_variables(self, fmt, scope)

### Function: __init__(self, suspended_frames_manager, py_db)

### Function: _register_variable(self, variable)

### Function: obtain_as_variable(self, name, value, evaluate_name, frame)

### Function: get_main_thread_id(self)

### Function: get_variable(self, variable_reference)

### Function: track(self, thread_id, frames_list, frame_custom_thread_id)

**Description:** :param thread_id:
    The thread id to be used for this frame.

:param FramesList frames_list:
    A list of frames to be tracked (the first is the topmost frame which is suspended at the given thread).

:param frame_custom_thread_id:
    If None this this is the id of the thread id for the custom frame (i.e.: coroutine).

### Function: untrack_all(self)

### Function: get_frames_list(self, thread_id)

### Function: find_frame(self, thread_id, frame_id)

### Function: create_thread_suspend_command(self, thread_id, stop_reason, message, trace_suspend_type, thread, additional_info)

### Function: __init__(self)

### Function: _get_tracker_for_variable_reference(self, variable_reference)

### Function: get_thread_id_for_variable_reference(self, variable_reference)

**Description:** We can't evaluate variable references values on any thread, only in the suspended
thread (the main reason for this is that in UI frameworks inspecting a UI object
from a different thread can potentially crash the application).

:param int variable_reference:
    The variable reference (can be either a frame id or a reference to a previously
    gotten variable).

:return str:
    The thread id for the thread to be used to inspect the given variable reference or
    None if the thread was already resumed.

### Function: get_frame_tracker(self, thread_id)

### Function: get_variable(self, variable_reference)

**Description:** :raises KeyError

### Function: get_frames_list(self, thread_id)

### Function: track_frames(self, py_db)

### Function: add_fake_frame(self, thread_id, frame_id, frame)

### Function: find_frame(self, thread_id, frame_id)
