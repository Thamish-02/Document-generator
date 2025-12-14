## AI Summary

A file named django_debug.py.


## Class: DjangoLineBreakpoint

## Class: _DjangoValidationInfo

### Function: add_line_breakpoint(pydb, type, canonical_normalized_filename, breakpoint_id, line, condition, expression, func_name, hit_condition, is_logpoint, add_breakpoint_result, on_changed_breakpoint_state)

### Function: after_breakpoints_consolidated(py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)

### Function: add_exception_breakpoint(pydb, type, exception)

### Function: _init_plugin_breaks(pydb)

### Function: remove_exception_breakpoint(pydb, exception_type, exception)

### Function: remove_all_exception_breakpoints(pydb)

### Function: get_breakpoints(pydb, breakpoint_type)

### Function: _inherits(cls)

### Function: _is_django_render_call(frame)

### Function: _is_django_context_get_call(frame)

### Function: _is_django_resolve_call(frame)

### Function: _is_django_suspended(thread)

### Function: suspend_django(py_db, thread, frame, cmd)

### Function: _find_django_render_frame(frame)

### Function: _read_file(filename)

### Function: _offset_to_line_number(text, offset)

### Function: _get_source_django_18_or_lower(frame)

### Function: _convert_to_str(s)

### Function: _get_template_original_file_name_from_frame(frame)

### Function: _get_template_line(frame)

## Class: DjangoTemplateFrame

## Class: DjangoTemplateSyntaxErrorFrame

### Function: change_variable(frame, attr, expression, default, scope)

### Function: _is_django_variable_does_not_exist_exception_break_context(frame)

### Function: _is_ignoring_failures(frame)

### Function: can_skip(py_db, frame)

### Function: required_events_breakpoint()

### Function: required_events_stepping()

### Function: has_exception_breaks(py_db)

### Function: has_line_breaks(py_db)

### Function: cmd_step_into(py_db, frame, event, info, thread, stop_info, stop)

### Function: cmd_step_over(py_db, frame, event, info, thread, stop_info, stop)

### Function: stop(py_db, frame, event, thread, stop_info, arg, step_cmd)

### Function: get_breakpoint(py_db, frame, event, info)

### Function: suspend(py_db, thread, frame, bp_type)

### Function: _get_original_filename_from_origin_in_parent_frame_locals(frame, parent_frame_name)

### Function: exception_break(py_db, frame, thread, arg, is_unwind)

### Function: __init__(self, canonical_normalized_filename, breakpoint_id, line, condition, func_name, expression, hit_condition, is_logpoint)

### Function: __str__(self)

### Function: _collect_valid_lines_in_template_uncached(self, template)

### Function: _get_lineno(self, node)

### Function: _iternodes(self, nodelist)

### Function: __init__(self, frame)

### Function: _collect_context(self, context)

### Function: _change_variable(self, name, value)

### Function: __init__(self, frame, original_filename, lineno, f_locals)
