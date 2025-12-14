## AI Summary

A file named pydevd_process_net_command_json.py.


### Function: _convert_rules_to_exclude_filters(rules, on_error)

## Class: IDMap

## Class: PyDevJsonCommandProcessor

### Function: __init__(self)

### Function: obtain_value(self, key)

### Function: obtain_key(self, value)

### Function: __init__(self, from_json)

### Function: process_net_command_json(self, py_db, json_contents, send_response)

**Description:** Processes a debug adapter protocol json command.

### Function: on_pydevdauthorize_request(self, py_db, request)

### Function: on_initialize_request(self, py_db, request)

### Function: on_configurationdone_request(self, py_db, request)

**Description:** :param ConfigurationDoneRequest request:

### Function: on_threads_request(self, py_db, request)

**Description:** :param ThreadsRequest request:

### Function: on_terminate_request(self, py_db, request)

**Description:** :param TerminateRequest request:

### Function: _request_terminate_process(self, py_db)

### Function: on_completions_request(self, py_db, request)

**Description:** :param CompletionsRequest request:

### Function: _resolve_remote_root(self, local_root, remote_root)

### Function: _set_debug_options(self, py_db, args, start_reason)

### Function: _send_process_event(self, py_db, start_method)

### Function: _handle_launch_or_attach_request(self, py_db, request, start_reason)

### Function: on_launch_request(self, py_db, request)

**Description:** :param LaunchRequest request:

### Function: on_attach_request(self, py_db, request)

**Description:** :param AttachRequest request:

### Function: on_pause_request(self, py_db, request)

**Description:** :param PauseRequest request:

### Function: on_continue_request(self, py_db, request)

**Description:** :param ContinueRequest request:

### Function: on_next_request(self, py_db, request)

**Description:** :param NextRequest request:

### Function: on_stepin_request(self, py_db, request)

**Description:** :param StepInRequest request:

### Function: on_stepintargets_request(self, py_db, request)

**Description:** :param StepInTargetsRequest request:

### Function: on_stepout_request(self, py_db, request)

**Description:** :param StepOutRequest request:

### Function: _get_hit_condition_expression(self, hit_condition)

**Description:** Following hit condition values are supported

* x or == x when breakpoint is hit x times
* >= x when breakpoint is hit more than or equal to x times
* % x when breakpoint is hit multiple of x times

Returns '@HIT@ == x' where @HIT@ will be replaced by number of hits

### Function: on_disconnect_request(self, py_db, request)

**Description:** :param DisconnectRequest request:

### Function: _verify_launch_or_attach_done(self, request)

### Function: on_setfunctionbreakpoints_request(self, py_db, request)

**Description:** :param SetFunctionBreakpointsRequest request:

### Function: on_setbreakpoints_request(self, py_db, request)

**Description:** :param SetBreakpointsRequest request:

### Function: _on_changed_breakpoint_state(self, py_db, source, breakpoint_id, result)

### Function: _create_breakpoint_from_add_breakpoint_result(self, py_db, source, breakpoint_id, result)

### Function: on_setexceptionbreakpoints_request(self, py_db, request)

**Description:** :param SetExceptionBreakpointsRequest request:

### Function: on_stacktrace_request(self, py_db, request)

**Description:** :param StackTraceRequest request:

### Function: on_exceptioninfo_request(self, py_db, request)

**Description:** :param ExceptionInfoRequest request:

### Function: on_scopes_request(self, py_db, request)

**Description:** Scopes are the top-level items which appear for a frame (so, we receive the frame id
and provide the scopes it has).

:param ScopesRequest request:

### Function: on_evaluate_request(self, py_db, request)

**Description:** :param EvaluateRequest request:

### Function: on_setexpression_request(self, py_db, request)

### Function: on_variables_request(self, py_db, request)

**Description:** Variables can be asked whenever some place returned a variables reference (so, it
can be a scope gotten from on_scopes_request, the result of some evaluation, etc.).

Note that in the DAP the variables reference requires a unique int... the way this works for
pydevd is that an instance is generated for that specific variable reference and we use its
id(instance) to identify it to make sure all items are unique (and the actual {id->instance}
is added to a dict which is only valid while the thread is suspended and later cleared when
the related thread resumes execution).

see: SuspendedFramesManager

:param VariablesRequest request:

### Function: on_setvariable_request(self, py_db, request)

### Function: on_modules_request(self, py_db, request)

### Function: on_source_request(self, py_db, request)

**Description:** :param SourceRequest request:

### Function: on_gototargets_request(self, py_db, request)

### Function: on_goto_request(self, py_db, request)

### Function: on_setdebuggerproperty_request(self, py_db, request)

### Function: on_pydevdsysteminfo_request(self, py_db, request)

### Function: on_setpydevdsourcemap_request(self, py_db, request)

### Function: on_resumed()

### Function: get_variable_presentation(setting, default)

### Function: on_request(py_db, request)
