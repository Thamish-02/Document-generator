## AI Summary

A file named _pydevd_sys_monitoring.py.


### Function: _notify_skipped_step_in_because_of_filters(py_db, frame)

### Function: _get_bootstrap_frame(depth)

### Function: _get_unhandled_exception_frame(exc, depth)

## Class: ThreadInfo

## Class: _DeleteDummyThreadOnDel

**Description:** Helper class to remove a dummy thread from threading._active on __del__.

### Function: _create_thread_info(depth)

## Class: FuncCodeInfo

### Function: _get_thread_info(create, depth)

**Description:** Provides thread-related info.

May return None if the thread is still not active.

## Class: _CodeLineInfo

### Function: _get_code_line_info(code_obj, _cache)

### Function: _get_func_code_info(code_obj, frame_or_depth)

**Description:** Provides code-object related info.

Note that it contains informations on the breakpoints for a given function.
If breakpoints change a new FuncCodeInfo instance will be created.

Note that this can be called by any thread.

### Function: _enable_line_tracing(code)

### Function: _enable_return_tracing(code)

### Function: disable_code_tracing(code)

### Function: enable_code_tracing(thread_ident, code, frame)

**Description:** Note: this must enable code tracing for the given code/frame.

The frame can be from any thread!

:return: Whether code tracing was added in this function to the given code.

### Function: reset_thread_local_info()

**Description:** Resets the thread local info TLS store for use after a fork().

### Function: _enable_code_tracing(py_db, additional_info, func_code_info, code, frame, warn_on_filtered_out)

**Description:** :return: Whether code tracing was added in this function to the given code.

### Function: _enable_step_tracing(py_db, code, step_cmd, info, frame)

## Class: _TryExceptContainerObj

**Description:** A dumb container object just to contain the try..except info when needed. Meant to be
persistent among multiple PyDBFrames to the same code object.

### Function: _unwind_event(code, instruction, exc)

### Function: _raise_event(code, instruction, exc)

**Description:** The way this should work is the following: when the user is using
pydevd to do the launch and we're on a managed stack, we should consider
unhandled only if it gets into a pydevd. If it's a thread, if it stops
inside the threading and if it's an unmanaged thread (i.e.: QThread)
then stop if it doesn't have a back frame.

Note: unlike other events, this one is global and not per-code (so,
it cannot be individually enabled/disabled for a given code object).

### Function: get_func_name(frame)

### Function: _show_return_values(frame, arg)

### Function: _remove_return_values(py_db, frame)

### Function: _return_event(code, instruction, retval)

### Function: _enable_code_tracing_for_frame_and_parents(thread_info, frame)

### Function: _stop_on_return(py_db, thread_info, info, step_cmd, frame, retval)

### Function: _stop_on_breakpoint(py_db, thread_info, stop_reason, bp, frame, new_frame, stop, stop_on_plugin_breakpoint, bp_type)

**Description:** :param bp: the breakpoint hit (additional conditions will be checked now).
:param frame: the actual frame
:param new_frame: either the actual frame or the frame provided by the plugins.
:param stop: whether we should do a regular line breakpoint.
:param stop_on_plugin_breakpoint: whether we should stop in a plugin breakpoint.

:return:
    True if the breakpoint was suspended inside this function and False otherwise.
    Note that even if False is returned, it's still possible

### Function: _plugin_stepping(py_db, step_cmd, event, frame, thread_info)

### Function: _jump_event(code, from_offset, to_offset)

### Function: _line_event(code, line)

### Function: _internal_line_event(func_code_info, frame, line)

### Function: _start_method_event(code, instruction_offset)

### Function: _ensure_monitoring()

### Function: start_monitoring(all_threads)

### Function: stop_monitoring(all_threads)

### Function: update_monitor_events(suspend_requested)

**Description:** This should be called when breakpoints change.

:param suspend: means the user requested threads to be suspended

### Function: restart_events()

### Function: _is_same_frame(info, target_frame, current_frame)

### Function: _do_wait_suspend(py_db, thread_info, frame, event, arg)

### Function: __init__(self, thread, thread_ident, trace, additional_info)

### Function: is_thread_alive(self)

### Function: __init__(self, dummy_thread)

### Function: __del__(self)

### Function: __init__(self)

### Function: get_line_of_offset(self, offset)

### Function: __init__(self, line_to_offset, first_line, last_line)

### Function: __init__(self, try_except_infos)

### Function: get_smart_step_into_variant_from_frame_offset()
