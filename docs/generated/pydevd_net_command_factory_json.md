## AI Summary

A file named pydevd_net_command_factory_json.py.


## Class: ModulesManager

## Class: NetCommandFactoryJson

**Description:** Factory for commands which will provide messages as json (they should be
similar to the debug adapter where possible, although some differences
are currently Ok).

Note that it currently overrides the xml version so that messages
can be done one at a time (any message not overridden will currently
use the xml version) -- after having all messages handled, it should
no longer use NetCommandFactory as the base class.

### Function: __init__(self)

### Function: track_module(self, filename_in_utf8, module_name, frame)

**Description:** :return list(NetCommand):
    Returns a list with the module events to be sent.

### Function: get_modules_info(self)

**Description:** :return list(Module)

### Function: __init__(self)

### Function: make_version_message(self, seq)

### Function: make_protocol_set_message(self, seq)

### Function: make_thread_created_message(self, thread)

### Function: make_custom_frame_created_message(self, frame_id, frame_description)

### Function: make_thread_killed_message(self, tid)

### Function: make_list_threads_message(self, py_db, seq)

### Function: make_get_completions_message(self, seq, completions, qualifier, start)

### Function: _format_frame_name(self, fmt, initial_name, module_name, line, path)

### Function: make_get_thread_stack_message(self, py_db, seq, thread_id, topmost_frame, fmt, must_be_suspended, start_frame, levels)

### Function: make_warning_message(self, msg)

### Function: make_io_message(self, msg, ctx)

### Function: make_console_message(self, msg)

### Function: make_thread_suspend_single_notification(self, py_db, thread_id, thread, stop_reason)

### Function: make_thread_resume_single_notification(self, thread_id)

### Function: make_set_next_stmnt_status_message(self, seq, is_success, exception_msg)

### Function: make_send_curr_exception_trace_message(self)

### Function: make_send_curr_exception_trace_proceeded_message(self)

### Function: make_send_breakpoint_exception_message(self)

### Function: make_process_created_message(self)

### Function: make_process_about_to_be_replaced_message(self)

### Function: make_thread_suspend_message(self, py_db, thread_id, frames_list, stop_reason, message, trace_suspend_type, thread, info)

### Function: make_thread_run_message(self, py_db, thread_id, reason)

### Function: make_reloaded_code_message(self)

### Function: make_input_requested_message(self, started)

### Function: make_skipped_step_in_because_of_filters(self, py_db, frame)

### Function: make_evaluation_timeout_msg(self, py_db, expression, curr_thread)

### Function: make_exit_command(self, py_db)

### Function: after_send(socket)
