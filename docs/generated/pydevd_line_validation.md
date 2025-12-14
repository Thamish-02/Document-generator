## AI Summary

A file named pydevd_line_validation.py.


## Class: LineBreakpointWithLazyValidation

## Class: ValidationInfo

### Function: __init__(self)

### Function: __init__(self)

### Function: _collect_valid_lines_in_template(self, template)

### Function: _collect_valid_lines_in_template_uncached(self, template)

### Function: verify_breakpoints(self, py_db, canonical_normalized_filename, template_breakpoints_for_file, template)

**Description:** This function should be called whenever a rendering is detected.

:param str canonical_normalized_filename:
:param dict[int:LineBreakpointWithLazyValidation] template_breakpoints_for_file:

### Function: verify_breakpoints_from_template_cached_lines(self, py_db, canonical_normalized_filename, template_breakpoints_for_file)

**Description:** This is used when the lines are already available (if just the template is available,
`verify_breakpoints` should be used instead).

### Function: _verify_breakpoints_with_lines_collected(self, py_db, canonical_normalized_filename, template_breakpoints_for_file, valid_lines_frozenset, sorted_lines)
