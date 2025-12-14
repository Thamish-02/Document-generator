## AI Summary

A file named pydevd_net_command_factory_xml.py.


## Class: NetCommandFactory

### Function: __init__(self)

### Function: _thread_to_xml(self, thread)

**Description:** thread information as XML

### Function: make_error_message(self, seq, text)

### Function: make_protocol_set_message(self, seq)

### Function: make_thread_created_message(self, thread)

### Function: make_process_created_message(self)

### Function: make_process_about_to_be_replaced_message(self)

### Function: make_show_cython_warning_message(self)

### Function: make_custom_frame_created_message(self, frame_id, frame_description)

### Function: make_list_threads_message(self, py_db, seq)

**Description:** returns thread listing as XML

### Function: make_get_thread_stack_message(self, py_db, seq, thread_id, topmost_frame, fmt, must_be_suspended, start_frame, levels)

**Description:** Returns thread stack as XML.

:param must_be_suspended: If True and the thread is not suspended, returns None.

### Function: make_variable_changed_message(self, seq, payload)

### Function: make_warning_message(self, msg)

### Function: make_console_message(self, msg)

### Function: make_io_message(self, msg, ctx)

**Description:** @param msg: the message to pass to the debug server
@param ctx: 1 for stdio 2 for stderr

### Function: make_version_message(self, seq)

### Function: make_thread_killed_message(self, tid)

### Function: _iter_visible_frames_info(self, py_db, frames_list, flatten_chained)

### Function: make_thread_stack_str(self, py_db, frames_list)

### Function: make_thread_suspend_str(self, py_db, thread_id, frames_list, stop_reason, message, trace_suspend_type)

**Description:** :return tuple(str,str):
    Returns tuple(thread_suspended_str, thread_stack_str).

    i.e.:
    (
        '''
            <xml>
                <thread id="id" stop_reason="reason">
                    <frame id="id" name="functionName " file="file" line="line">
                    </frame>
                </thread>
            </xml>
        '''
        ,
        '''
        <frame id="id" name="functionName " file="file" line="line">
        </frame>
        '''
    )

### Function: make_thread_suspend_message(self, py_db, thread_id, frames_list, stop_reason, message, trace_suspend_type, thread, additional_info)

### Function: make_thread_suspend_single_notification(self, py_db, thread_id, thread, stop_reason)

### Function: make_thread_resume_single_notification(self, thread_id)

### Function: make_thread_run_message(self, py_db, thread_id, reason)

### Function: make_get_variable_message(self, seq, payload)

### Function: make_get_array_message(self, seq, payload)

### Function: make_get_description_message(self, seq, payload)

### Function: make_get_frame_message(self, seq, payload)

### Function: make_evaluate_expression_message(self, seq, payload)

### Function: make_get_completions_message(self, seq, completions, qualifier, start)

### Function: make_get_file_contents(self, seq, payload)

### Function: make_reloaded_code_message(self, seq, reloaded_ok)

### Function: make_send_breakpoint_exception_message(self, seq, payload)

### Function: _make_send_curr_exception_trace_str(self, py_db, thread_id, exc_type, exc_desc, trace_obj)

### Function: make_send_curr_exception_trace_message(self, py_db, seq, thread_id, curr_frame_id, exc_type, exc_desc, trace_obj)

### Function: make_get_exception_details_message(self, py_db, seq, thread_id, topmost_frame)

**Description:** Returns exception details as XML

### Function: make_send_curr_exception_trace_proceeded_message(self, seq, thread_id)

### Function: make_send_console_message(self, seq, payload)

### Function: make_custom_operation_message(self, seq, payload)

### Function: make_load_source_message(self, seq, source)

### Function: make_load_source_from_frame_id_message(self, seq, source)

### Function: make_show_console_message(self, py_db, thread_id, frame)

### Function: make_input_requested_message(self, started)

### Function: make_set_next_stmnt_status_message(self, seq, is_success, exception_msg)

### Function: make_load_full_value_message(self, seq, payload)

### Function: make_get_next_statement_targets_message(self, seq, payload)

### Function: make_skipped_step_in_because_of_filters(self, py_db, frame)

### Function: make_evaluation_timeout_msg(self, py_db, expression, thread)

### Function: make_exit_command(self, py_db)
