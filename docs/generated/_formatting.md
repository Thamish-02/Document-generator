## AI Summary

A file named _formatting.py.


### Function: _format_final_exc_line(etype, value)

### Function: _safe_string(value, what, func)

## Class: _ExceptionPrintContext

### Function: exceptiongroup_excepthook(etype, value, tb)

## Class: PatchedTracebackException

### Function: format_exception_only(__exc)

### Function: _(__exc, value)

### Function: format_exception(__exc, limit, chain)

### Function: _(__exc, value, tb, limit, chain)

### Function: print_exception(__exc, limit, file, chain)

### Function: _(__exc, value, tb, limit, file, chain)

### Function: print_exc(limit, file, chain)

### Function: _substitution_cost(ch_a, ch_b)

### Function: _compute_suggestion_error(exc_value, tb)

### Function: _levenshtein_distance(a, b, max_cost)

### Function: __init__(self)

### Function: indent(self)

### Function: emit(self, text_gen, margin_char)

### Function: __init__(self, exc_type, exc_value, exc_traceback)

### Function: format(self)

### Function: format_exception_only(self)

**Description:** Format the exception part of the traceback.
The return value is a generator of strings, each ending in a newline.
Normally, the generator emits a single string; however, for
SyntaxError exceptions, it emits several lines that (when
printed) display detailed information about where the syntax
error occurred.
The message indicating which exception occurred is always the last
string in the output.
