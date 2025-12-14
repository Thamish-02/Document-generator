## AI Summary

A file named pydevd_api.py.


## Class: PyDevdAPI

### Function: _list_ppid_and_pid()

### Function: _get_code_lines(code)

## Class: VariablePresentation

### Function: run(self, py_db)

### Function: notify_initialize(self, py_db)

### Function: notify_configuration_done(self, py_db)

### Function: notify_disconnect(self, py_db)

### Function: set_protocol(self, py_db, seq, protocol)

### Function: set_ide_os_and_breakpoints_by(self, py_db, seq, ide_os, breakpoints_by)

**Description:** :param ide_os: 'WINDOWS' or 'UNIX'
:param breakpoints_by: 'ID' or 'LINE'

### Function: set_ide_os(self, ide_os)

**Description:** :param ide_os: 'WINDOWS' or 'UNIX'

### Function: set_gui_event_loop(self, py_db, gui_event_loop)

### Function: send_error_message(self, py_db, msg)

### Function: set_show_return_values(self, py_db, show_return_values)

### Function: list_threads(self, py_db, seq)

### Function: request_suspend_thread(self, py_db, thread_id)

### Function: set_enable_thread_notifications(self, py_db, enable)

**Description:** When disabled, no thread notifications (for creation/removal) will be
issued until it's re-enabled.

Note that when it's re-enabled, a creation notification will be sent for
all existing threads even if it was previously sent (this is meant to
be used on disconnect/reconnect).

### Function: request_disconnect(self, py_db, resume_threads)

### Function: request_resume_thread(self, thread_id)

### Function: request_completions(self, py_db, seq, thread_id, frame_id, act_tok, line, column)

### Function: request_stack(self, py_db, seq, thread_id, fmt, timeout, start_frame, levels)

### Function: request_exception_info_json(self, py_db, request, thread_id, thread, max_frames)

### Function: request_step(self, py_db, thread_id, step_cmd_id)

### Function: request_smart_step_into(self, py_db, seq, thread_id, offset, child_offset)

### Function: request_smart_step_into_by_func_name(self, py_db, seq, thread_id, line, func_name)

### Function: request_set_next(self, py_db, seq, thread_id, set_next_cmd_id, original_filename, line, func_name)

**Description:** set_next_cmd_id may actually be one of:

CMD_RUN_TO_LINE
CMD_SET_NEXT_STATEMENT

CMD_SMART_STEP_INTO -- note: request_smart_step_into is preferred if it's possible
                       to work with bytecode offset.

:param Optional[str] original_filename:
    If available, the filename may be source translated, otherwise no translation will take
    place (the set next just needs the line afterwards as it executes locally, but for
    the Jupyter integration, the source mapping may change the actual lines and not only
    the filename).

### Function: request_reload_code(self, py_db, seq, module_name, filename)

**Description:** :param seq: if -1 no message will be sent back when the reload is done.

Note: either module_name or filename may be None (but not both at the same time).

### Function: request_change_variable(self, py_db, seq, thread_id, frame_id, scope, attr, value)

**Description:** :param scope: 'FRAME' or 'GLOBAL'

### Function: request_get_variable(self, py_db, seq, thread_id, frame_id, scope, attrs)

**Description:** :param scope: 'FRAME' or 'GLOBAL'

### Function: request_get_array(self, py_db, seq, roffset, coffset, rows, cols, fmt, thread_id, frame_id, scope, attrs)

### Function: request_load_full_value(self, py_db, seq, thread_id, frame_id, vars)

### Function: request_get_description(self, py_db, seq, thread_id, frame_id, expression)

### Function: request_get_frame(self, py_db, seq, thread_id, frame_id)

### Function: to_str(self, s)

**Description:** -- in py3 raises an error if it's not str already.

### Function: filename_to_str(self, filename)

**Description:** -- in py3 raises an error if it's not str already.

### Function: filename_to_server(self, filename)

## Class: _DummyFrame

**Description:** Dummy frame to be used with PyDB.apply_files_filter (as we don't really have the
related frame as breakpoints are added before execution).

## Class: _AddBreakpointResult

### Function: add_breakpoint(self, py_db, original_filename, breakpoint_type, breakpoint_id, line, condition, func_name, expression, suspend_policy, hit_condition, is_logpoint, adjust_line, on_changed_breakpoint_state)

**Description:** :param str original_filename:
    Note: must be sent as it was received in the protocol. It may be translated in this
    function and its final value will be available in the returned _AddBreakpointResult.

:param str breakpoint_type:
    One of: 'python-line', 'django-line', 'jinja2-line'.

:param int breakpoint_id:

:param int line:
    Note: it's possible that a new line was actually used. If that's the case its
    final value will be available in the returned _AddBreakpointResult.

:param condition:
    Either None or the condition to activate the breakpoint.

:param str func_name:
    If "None" (str), may hit in any context.
    Empty string will hit only top level.
    Any other value must match the scope of the method to be matched.

:param str expression:
    None or the expression to be evaluated.

:param suspend_policy:
    Either "NONE" (to suspend only the current thread when the breakpoint is hit) or
    "ALL" (to suspend all threads when a breakpoint is hit).

:param str hit_condition:
    An expression where `@HIT@` will be replaced by the number of hits.
    i.e.: `@HIT@ == x` or `@HIT@ >= x`

:param bool is_logpoint:
    If True and an expression is passed, pydevd will create an io message command with the
    result of the evaluation.

:param bool adjust_line:
    If True, the breakpoint line should be adjusted if the current line doesn't really
    match an executable line (if possible).

:param callable on_changed_breakpoint_state:
    This is called when something changed internally on the breakpoint after it was initially
    added (for instance, template file_to_line_to_breakpoints could be signaled as invalid initially and later
    when the related template is loaded, if the line is valid it could be marked as valid).

    The signature for the callback should be:
        on_changed_breakpoint_state(breakpoint_id: int, add_breakpoint_result: _AddBreakpointResult)

        Note that the add_breakpoint_result should not be modified by the callback (the
        implementation may internally reuse the same instance multiple times).

:return _AddBreakpointResult:

### Function: reapply_breakpoints(self, py_db)

**Description:** Reapplies all the received breakpoints as they were received by the API (so, new
translations are applied).

### Function: remove_all_breakpoints(self, py_db, received_filename)

**Description:** Removes all the breakpoints from a given file or from all files if received_filename == '*'.

:param str received_filename:
    Note: must be sent as it was received in the protocol. It may be translated in this
    function.

### Function: remove_breakpoint(self, py_db, received_filename, breakpoint_type, breakpoint_id)

**Description:** :param str received_filename:
    Note: must be sent as it was received in the protocol. It may be translated in this
    function.

:param str breakpoint_type:
    One of: 'python-line', 'django-line', 'jinja2-line'.

:param int breakpoint_id:

### Function: set_function_breakpoints(self, py_db, function_breakpoints)

### Function: request_exec_or_evaluate(self, py_db, seq, thread_id, frame_id, expression, is_exec, trim_if_too_big, attr_to_set_result)

### Function: request_exec_or_evaluate_json(self, py_db, request, thread_id)

### Function: request_set_expression_json(self, py_db, request, thread_id)

### Function: request_console_exec(self, py_db, seq, thread_id, frame_id, expression)

### Function: request_load_source(self, py_db, seq, filename)

**Description:** :param str filename:
    Note: must be sent as it was received in the protocol. It may be translated in this
    function.

### Function: get_decompiled_source_from_frame_id(self, py_db, frame_id)

**Description:** :param py_db:
:param frame_id:
:throws Exception:
    If unable to get the frame in the currently paused frames or if some error happened
    when decompiling.

### Function: request_load_source_from_frame_id(self, py_db, seq, frame_id)

### Function: add_python_exception_breakpoint(self, py_db, exception, condition, expression, notify_on_handled_exceptions, notify_on_unhandled_exceptions, notify_on_user_unhandled_exceptions, notify_on_first_raise_only, ignore_libraries)

### Function: add_plugins_exception_breakpoint(self, py_db, breakpoint_type, exception)

### Function: remove_python_exception_breakpoint(self, py_db, exception)

### Function: remove_plugins_exception_breakpoint(self, py_db, exception_type, exception)

### Function: remove_all_exception_breakpoints(self, py_db)

### Function: set_project_roots(self, py_db, project_roots)

**Description:** :param str project_roots:

### Function: set_stepping_resumes_all_threads(self, py_db, stepping_resumes_all_threads)

### Function: set_exclude_filters(self, py_db, exclude_filters)

**Description:** :param list(PyDevdAPI.ExcludeFilter) exclude_filters:

### Function: set_use_libraries_filter(self, py_db, use_libraries_filter)

### Function: request_get_variable_json(self, py_db, request, thread_id)

**Description:** :param VariablesRequest request:

### Function: request_change_variable_json(self, py_db, request, thread_id)

**Description:** :param SetVariableRequest request:

### Function: set_dont_trace_start_end_patterns(self, py_db, start_patterns, end_patterns)

### Function: stop_on_entry(self)

### Function: set_ignore_system_exit_codes(self, py_db, ignore_system_exit_codes)

### Function: set_source_mapping(self, py_db, source_filename, mapping)

**Description:** :param str source_filename:
    The filename for the source mapping (bytes on py2 and str on py3).
    This filename will be made absolute in this function.

:param list(SourceMappingEntry) mapping:
    A list with the source mapping entries to be applied to the given filename.

:return str:
    An error message if it was not possible to set the mapping or an empty string if
    everything is ok.

### Function: set_variable_presentation(self, py_db, variable_presentation)

### Function: get_ppid(self)

**Description:** Provides the parent pid (even for older versions of Python on Windows).

### Function: _get_windows_ppid(self)

### Function: _terminate_child_processes_windows(self, dont_terminate_child_pids)

### Function: _terminate_child_processes_linux_and_mac(self, dont_terminate_child_pids)

### Function: _popen(self, cmdline)

### Function: _call(self, cmdline)

### Function: set_terminate_child_processes(self, py_db, terminate_child_processes)

### Function: set_terminate_keyboard_interrupt(self, py_db, terminate_keyboard_interrupt)

### Function: terminate_process(self, py_db)

**Description:** Terminates the current process (and child processes if the option to also terminate
child processes is enabled).

### Function: _terminate_if_commands_processed(self, py_db)

### Function: request_terminate_process(self, py_db)

### Function: setup_auto_reload_watcher(self, py_db, enable_auto_reload, watch_dirs, poll_target_time, exclude_patterns, include_patterns)

## Class: PROCESSENTRY32

### Function: _get_code_lines(code)

### Function: iterate()

### Function: __init__(self, special, function, class_, protected)

### Function: get_presentation(self, scope)

## Class: _DummyCode

### Function: __init__(self, filename)

### Function: __init__(self, breakpoint_id, translated_filename, translated_line, original_line)

### Function: custom_dont_trace_external_files(abs_path)

### Function: list_children_and_stop_forking(initial_pid, stop)

### Function: __init__(self, filename)
