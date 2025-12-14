## AI Summary

A file named pydevd_process_net_command.py.


## Class: _PyDevCommandProcessor

### Function: __init__(self)

### Function: process_net_command(self, py_db, cmd_id, seq, text)

**Description:** Processes a command received from the Java side

@param cmd_id: the id of the command
@param seq: the sequence of the command
@param text: the text received in the command

### Function: cmd_authenticate(self, py_db, cmd_id, seq, text)

### Function: cmd_run(self, py_db, cmd_id, seq, text)

### Function: cmd_list_threads(self, py_db, cmd_id, seq, text)

### Function: cmd_get_completions(self, py_db, cmd_id, seq, text)

### Function: cmd_get_thread_stack(self, py_db, cmd_id, seq, text)

### Function: cmd_set_protocol(self, py_db, cmd_id, seq, text)

### Function: cmd_thread_suspend(self, py_db, cmd_id, seq, text)

### Function: cmd_version(self, py_db, cmd_id, seq, text)

### Function: cmd_thread_run(self, py_db, cmd_id, seq, text)

### Function: _cmd_step(self, py_db, cmd_id, seq, text)

### Function: _cmd_set_next(self, py_db, cmd_id, seq, text)

### Function: cmd_smart_step_into(self, py_db, cmd_id, seq, text)

### Function: cmd_reload_code(self, py_db, cmd_id, seq, text)

### Function: cmd_change_variable(self, py_db, cmd_id, seq, text)

### Function: cmd_get_variable(self, py_db, cmd_id, seq, text)

### Function: cmd_get_array(self, py_db, cmd_id, seq, text)

### Function: cmd_show_return_values(self, py_db, cmd_id, seq, text)

### Function: cmd_load_full_value(self, py_db, cmd_id, seq, text)

### Function: cmd_get_description(self, py_db, cmd_id, seq, text)

### Function: cmd_get_frame(self, py_db, cmd_id, seq, text)

### Function: cmd_set_break(self, py_db, cmd_id, seq, text)

### Function: cmd_remove_break(self, py_db, cmd_id, seq, text)

### Function: _cmd_exec_or_evaluate_expression(self, py_db, cmd_id, seq, text)

### Function: cmd_console_exec(self, py_db, cmd_id, seq, text)

### Function: cmd_set_path_mapping_json(self, py_db, cmd_id, seq, text)

**Description:** :param text:
    Json text. Something as:

    {
        "pathMappings": [
            {
                "localRoot": "c:/temp",
                "remoteRoot": "/usr/temp"
            }
        ],
        "debug": true,
        "force": false
    }

### Function: cmd_set_py_exception_json(self, py_db, cmd_id, seq, text)

### Function: cmd_set_py_exception(self, py_db, cmd_id, seq, text)

### Function: _load_source(self, py_db, cmd_id, seq, text)

### Function: cmd_load_source_from_frame_id(self, py_db, cmd_id, seq, text)

### Function: cmd_set_property_trace(self, py_db, cmd_id, seq, text)

### Function: cmd_add_exception_break(self, py_db, cmd_id, seq, text)

### Function: cmd_remove_exception_break(self, py_db, cmd_id, seq, text)

### Function: cmd_add_django_exception_break(self, py_db, cmd_id, seq, text)

### Function: cmd_remove_django_exception_break(self, py_db, cmd_id, seq, text)

### Function: cmd_evaluate_console_expression(self, py_db, cmd_id, seq, text)

### Function: cmd_run_custom_operation(self, py_db, cmd_id, seq, text)

### Function: cmd_ignore_thrown_exception_at(self, py_db, cmd_id, seq, text)

### Function: cmd_enable_dont_trace(self, py_db, cmd_id, seq, text)

### Function: cmd_redirect_output(self, py_db, cmd_id, seq, text)

### Function: cmd_get_next_statement_targets(self, py_db, cmd_id, seq, text)

### Function: cmd_get_smart_step_into_variants(self, py_db, cmd_id, seq, text)

### Function: cmd_set_project_roots(self, py_db, cmd_id, seq, text)

### Function: cmd_thread_dump_to_stderr(self, py_db, cmd_id, seq, text)

### Function: cmd_stop_on_start(self, py_db, cmd_id, seq, text)

### Function: cmd_pydevd_json_config(self, py_db, cmd_id, seq, text)

### Function: cmd_get_exception_details(self, py_db, cmd_id, seq, text)

### Function: on_changed_breakpoint_state(breakpoint_id, add_breakpoint_result)
