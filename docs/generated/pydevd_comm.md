## AI Summary

A file named pydevd_comm.py.


## Class: ReaderThread

**Description:** reader thread reads and dispatches commands in an infinite loop

## Class: FSNotifyThread

## Class: WriterThread

**Description:** writer thread writes out the commands in an infinite loop

### Function: create_server_socket(host, port)

### Function: start_server(port)

**Description:** binds to a port, waits for the debugger to connect

### Function: start_client(host, port)

**Description:** connects to a host/port

## Class: InternalThreadCommand

**Description:** internal commands are generated/executed by the debugger.

The reason for their existence is that some commands have to be executed
on specific threads. These are the InternalThreadCommands that get
get posted to PyDB.

## Class: InternalThreadCommandForAnyThread

### Function: _send_io_message(py_db, s)

### Function: internal_reload_code(dbg, seq, module_name, filename)

## Class: InternalGetThreadStack

**Description:** This command will either wait for a given thread to be paused to get its stack or will provide
it anyways after a timeout (in which case the stack will be gotten but local variables won't
be available and it'll not be possible to interact with the frame as it's not actually
stopped in a breakpoint).

### Function: internal_step_in_thread(py_db, thread_id, cmd_id, set_additional_thread_info)

### Function: internal_smart_step_into(py_db, thread_id, offset, child_offset, set_additional_thread_info)

## Class: InternalSetNextStatementThread

### Function: internal_get_variable_json(py_db, request)

**Description:** :param VariablesRequest request:

## Class: InternalGetVariable

**Description:** gets the value of a variable

## Class: InternalGetArray

### Function: internal_change_variable(dbg, seq, thread_id, frame_id, scope, attr, value)

**Description:** Changes the value of a variable

### Function: internal_change_variable_json(py_db, request)

**Description:** The pydevd_vars.change_attr_expression(thread_id, frame_id, attr, value, dbg) can only
deal with changing at a frame level, so, currently changing the contents of something
in a different scope is currently not supported.

:param SetVariableRequest request:

### Function: _write_variable_response(py_db, request, value, success, message)

### Function: internal_get_frame(dbg, seq, thread_id, frame_id)

**Description:** Converts request into python variable

### Function: internal_get_smart_step_into_variants(dbg, seq, thread_id, frame_id, start_line, end_line, set_additional_thread_info)

### Function: internal_get_step_in_targets_json(dbg, seq, thread_id, frame_id, request, set_additional_thread_info)

### Function: internal_get_next_statement_targets(dbg, seq, thread_id, frame_id)

**Description:** gets the valid line numbers for use with set next statement

### Function: _evaluate_response(py_db, request, result, error_message)

### Function: internal_evaluate_expression_json(py_db, request, thread_id)

**Description:** :param EvaluateRequest request:

### Function: _evaluate_response_return_exception(py_db, request, exc_type, exc, initial_tb)

### Function: internal_evaluate_expression(dbg, seq, thread_id, frame_id, expression, is_exec, trim_if_too_big, attr_to_set_result)

**Description:** gets the value of a variable

### Function: _set_expression_response(py_db, request, error_message)

### Function: internal_set_expression_json(py_db, request, thread_id)

### Function: internal_get_completions(dbg, seq, thread_id, frame_id, act_tok, line, column)

**Description:** Note that if the column is >= 0, the act_tok is considered text and the actual
activation token/qualifier is computed in this command.

### Function: internal_get_description(dbg, seq, thread_id, frame_id, expression)

**Description:** Fetch the variable description stub from the debug console

### Function: build_exception_info_response(dbg, thread_id, thread, request_seq, set_additional_thread_info, iter_visible_frames_info, max_frames)

**Description:** :return ExceptionInfoResponse

### Function: internal_get_exception_details_json(dbg, request, thread_id, thread, max_frames, set_additional_thread_info, iter_visible_frames_info)

**Description:** Fetch exception details

## Class: InternalGetBreakpointException

**Description:** Send details of exception raised while evaluating conditional breakpoint

## Class: InternalSendCurrExceptionTrace

**Description:** Send details of the exception that was caught and where we've broken in.

## Class: InternalSendCurrExceptionTraceProceeded

**Description:** Send details of the exception that was caught and where we've broken in.

## Class: InternalEvaluateConsoleExpression

**Description:** Execute the given command in the debug console

## Class: InternalRunCustomOperation

**Description:** Run a custom command on an expression

## Class: InternalConsoleGetCompletions

**Description:** Fetch the completions in the debug console

## Class: InternalConsoleExec

**Description:** gets the value of a variable

## Class: InternalLoadFullValue

**Description:** Loads values asynchronously

## Class: AbstractGetValueAsyncThread

**Description:** Abstract class for a thread, which evaluates values for async variables

## Class: GetValueAsyncThreadDebug

**Description:** A thread for evaluation async values, which returns result for debugger
Create message and send it via writer thread

## Class: GetValueAsyncThreadConsole

**Description:** A thread for evaluation async values, which returns result for Console
Send result directly to Console's server

### Function: __init__(self, sock, py_db, PyDevJsonCommandProcessor, process_net_command, terminate_on_socket_close)

### Function: _from_json(self, json_msg, update_ids_from_dap)

### Function: _on_dict_loaded(self, dct)

### Function: do_kill_pydev_thread(self)

### Function: _read(self, size)

### Function: _read_line(self)

### Function: _on_run(self)

### Function: _terminate_on_socket_close(self)

### Function: process_command(self, cmd_id, seq, text)

### Function: __init__(self, py_db, api, watch_dirs)

### Function: _on_run(self)

### Function: do_kill_pydev_thread(self)

### Function: __init__(self, sock, py_db, terminate_on_socket_close)

### Function: add_command(self, cmd)

**Description:** cmd is NetCommand

### Function: _on_run(self)

**Description:** just loop and write responses

### Function: empty(self)

### Function: do_kill_pydev_thread(self)

### Function: __init__(self, thread_id, method)

### Function: can_be_executed_by(self, thread_id)

**Description:** By default, it must be in the same thread to be executed

### Function: do_it(self, dbg)

### Function: __str__(self)

### Function: __init__(self, thread_id, method)

### Function: can_be_executed_by(self, thread_id)

### Function: do_it(self, dbg)

### Function: __init__(self, seq, thread_id, py_db, set_additional_thread_info, fmt, timeout, start_frame, levels)

### Function: can_be_executed_by(self, _thread_id)

### Function: do_it(self, dbg)

### Function: __init__(self, thread_id, cmd_id, line, func_name, seq)

**Description:** cmd_id may actually be one of:

CMD_RUN_TO_LINE
CMD_SET_NEXT_STATEMENT
CMD_SMART_STEP_INTO

### Function: do_it(self, dbg)

### Function: __init__(self, seq, thread_id, frame_id, scope, attrs)

### Function: do_it(self, dbg)

**Description:** Converts request into python variable

### Function: __init__(self, seq, roffset, coffset, rows, cols, format, thread_id, frame_id, scope, attrs)

### Function: do_it(self, dbg)

### Function: __init__(self, thread_id, exc_type, stacktrace)

### Function: do_it(self, dbg)

### Function: __init__(self, thread_id, arg, curr_frame_id)

**Description:** :param arg: exception type, description, traceback object

### Function: do_it(self, dbg)

### Function: __init__(self, thread_id)

### Function: do_it(self, dbg)

### Function: __init__(self, seq, thread_id, frame_id, line, buffer_output)

### Function: do_it(self, dbg)

**Description:** Create an XML for console output, error and more (true/false)
<xml>
    <output message=output_message></output>
    <error message=error_message></error>
    <more>true/false</more>
</xml>

### Function: __init__(self, seq, thread_id, frame_id, scope, attrs, style, encoded_code_or_file, fnname)

### Function: do_it(self, dbg)

### Function: __init__(self, seq, thread_id, frame_id, act_tok)

### Function: do_it(self, dbg)

**Description:** Get completions and write back to the client

### Function: __init__(self, seq, thread_id, frame_id, expression)

### Function: init_matplotlib_in_debug_console(self, py_db)

### Function: do_it(self, py_db)

### Function: __init__(self, seq, thread_id, frame_id, vars)

### Function: do_it(self, dbg)

**Description:** Starts a thread that will load values asynchronously

### Function: __init__(self, py_db, frame_accessor, seq, var_objects)

### Function: send_result(self, xml)

### Function: _on_run(self)

### Function: send_result(self, xml)

### Function: send_result(self, xml)

### Function: __create_frame()
