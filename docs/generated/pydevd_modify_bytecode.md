## AI Summary

A file named pydevd_modify_bytecode.py.


## Class: DebugHelper

### Function: _get_code_line_info(code_obj)

### Function: get_instructions_to_add(stop_at_line, _pydev_stop_at_break, _pydev_needs_stop_at_break)

**Description:** This is the bytecode for something as:

    if _pydev_needs_stop_at_break():
        _pydev_stop_at_break()

but with some special handling for lines.

## Class: _Node

## Class: _HelperBytecodeList

**Description:** A helper double-linked list to make the manipulation a bit easier (so that we don't need
to keep track of indices that change) and performant (because adding multiple items to
the middle of a regular list isn't ideal).

### Function: insert_pydevd_breaks(code_to_modify, breakpoint_lines, code_line_info, _pydev_stop_at_break, _pydev_needs_stop_at_break)

**Description:** Inserts pydevd programmatic breaks into the code (at the given lines).

:param breakpoint_lines: set with the lines where we should add breakpoints.
:return: tuple(boolean flag whether insertion was successful, modified code).

### Function: __init__(self)

### Function: _get_filename(self, op_number, prefix)

### Function: write_bytecode(self, b, op_number, prefix)

### Function: write_dis(self, code_to_modify, op_number, prefix)

### Function: __init__(self, data)

### Function: append(self, data)

### Function: prepend(self, data)

### Function: __init__(self, lst)

### Function: append(self, data)

### Function: head(self)

### Function: tail(self)

### Function: __iter__(self)
