## AI Summary

A file named pydevd_frame_utils.py.


## Class: Frame

## Class: FCode

### Function: add_exception_to_frame(frame, exception_info)

### Function: remove_exception_from_frame(frame)

### Function: just_raised(trace)

### Function: short_tb(exc_tb)

### Function: short_frame(frame)

### Function: short_stack(frame)

### Function: ignore_exception_trace(trace)

### Function: cached_call(obj, func)

## Class: _LineColInfo

### Function: _utf8_byte_offset_to_character_offset(s, offset)

### Function: _extract_caret_anchors_in_bytes_from_line_segment(segment)

## Class: FramesList

## Class: _DummyFrameWrapper

### Function: create_frames_list_from_exception_cause(trace_obj, frame, exc_type, exc_desc, memo)

### Function: create_frames_list_from_traceback(trace_obj, frame, exc_type, exc_desc, exception_type)

**Description:** :param trace_obj:
    This is the traceback from which the list should be created.

:param frame:
    This is the first frame to be considered (i.e.: topmost frame). If None is passed, all
    the frames from the traceback are shown (so, None should be passed for unhandled exceptions).

:param exception_type:
    If this is an unhandled exception or user unhandled exception, we'll not trim the stack to create from the passed
    frame, rather, we'll just mark the frame in the frames list.

### Function: create_frames_list_from_frame(frame)

### Function: __init__(self, f_back, f_fileno, f_code, f_locals, f_globals, f_trace)

### Function: __init__(self, name, filename)

### Function: co_lines(self)

### Function: __init__(self, lineno, end_lineno, colno, end_colno)

### Function: map_columns_to_line(self, original_line)

**Description:** The columns internally are actually based on bytes.

Also, the position isn't always the ideal one as the start may not be
what we want (if the user has many subscripts in the line the start
will always be the same and only the end would change).
For more details see:
https://github.com/microsoft/debugpy/issues/1099#issuecomment-1303403995

So, this function maps the start/end columns to the position to be shown in the editor.

### Function: __init__(self)

### Function: append(self, frame)

### Function: last_frame(self)

### Function: __len__(self)

### Function: __iter__(self)

### Function: __repr__(self)

### Function: __init__(self, frame, f_lineno, f_back)

### Function: f_locals(self)

### Function: f_globals(self)

### Function: __str__(self)

### Function: _get_code_position(code, instruction_index)

### Function: _get_line_col_info_from_tb(tb)

### Function: _get_line_col_info_from_tb(tb)
