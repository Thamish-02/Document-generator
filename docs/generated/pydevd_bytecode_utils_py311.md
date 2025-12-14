## AI Summary

A file named pydevd_bytecode_utils_py311.py.


### Function: _is_inside(item_pos, container_pos)

### Function: _get_smart_step_into_targets(code)

### Function: calculate_smart_step_into_variants(frame, start_line, end_line, base)

**Description:** Calculate smart step into variants for the given line range.
:param frame:
:type frame: :py:class:`types.FrameType`
:param start_line:
:param end_line:
:return: A list of call names from the first to the last.
:note: it's guaranteed that the offsets appear in order.
:raise: :py:class:`RuntimeError` if failed to parse the bytecode or if dis cannot be used.
