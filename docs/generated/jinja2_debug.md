## AI Summary

A file named jinja2_debug.py.


## Class: Jinja2LineBreakpoint

## Class: _Jinja2ValidationInfo

### Function: add_line_breakpoint(pydb, type, canonical_normalized_filename, breakpoint_id, line, condition, expression, func_name, hit_condition, is_logpoint, add_breakpoint_result, on_changed_breakpoint_state)

### Function: after_breakpoints_consolidated(py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)

### Function: add_exception_breakpoint(pydb, type, exception)

### Function: _init_plugin_breaks(pydb)

### Function: remove_all_exception_breakpoints(pydb)

### Function: remove_exception_breakpoint(pydb, exception_type, exception)

### Function: get_breakpoints(pydb, breakpoint_type)

### Function: _is_jinja2_render_call(frame)

### Function: _suspend_jinja2(pydb, thread, frame, cmd, message)

### Function: _is_jinja2_suspended(thread)

### Function: _is_jinja2_context_call(frame)

### Function: _is_jinja2_internal_function(frame)

### Function: _find_jinja2_render_frame(frame)

## Class: Jinja2TemplateFrame

## Class: Jinja2TemplateSyntaxErrorFrame

### Function: change_variable(frame, attr, expression, default, scope)

### Function: _is_missing(item)

### Function: _find_render_function_frame(frame)

### Function: _get_jinja2_template_debug_info(frame)

### Function: _get_frame_lineno_mapping(jinja_template)

**Description:** :rtype: list(tuple(int,int))
:return: list((original_line, line_in_frame))

### Function: _get_jinja2_template_line(frame)

### Function: _convert_to_str(s)

### Function: _get_jinja2_template_original_filename(frame)

### Function: has_exception_breaks(py_db)

### Function: has_line_breaks(py_db)

### Function: can_skip(pydb, frame)

### Function: required_events_breakpoint()

### Function: required_events_stepping()

### Function: cmd_step_into(pydb, frame, event, info, thread, stop_info, stop)

### Function: cmd_step_over(pydb, frame, event, info, thread, stop_info, stop)

### Function: stop(pydb, frame, event, thread, stop_info, arg, step_cmd)

### Function: get_breakpoint(py_db, frame, event, info)

### Function: suspend(pydb, thread, frame, bp_type)

### Function: exception_break(pydb, frame, thread, arg, is_unwind)

### Function: __init__(self, canonical_normalized_filename, breakpoint_id, line, condition, func_name, expression, hit_condition, is_logpoint)

### Function: __str__(self)

### Function: _collect_valid_lines_in_template_uncached(self, template)

### Function: __init__(self, frame, original_filename, template_lineno)

### Function: _get_real_var_name(self, orig_name)

### Function: collect_context(self, frame)

### Function: _change_variable(self, frame, name, value)

### Function: __init__(self, frame, exception_cls_name, filename, lineno, f_locals)
