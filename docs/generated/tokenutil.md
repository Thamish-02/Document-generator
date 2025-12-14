## AI Summary

A file named tokenutil.py.


### Function: generate_tokens(readline)

**Description:** wrap generate_tkens to catch EOF errors

### Function: generate_tokens_catch_errors(readline, extra_errors_to_catch)

### Function: line_at_cursor(cell, cursor_pos)

**Description:** Return the line in a cell at a given cursor position

Used for calling line-based APIs that don't support multi-line input, yet.

Parameters
----------
cell : str
    multiline block of text
cursor_pos : integer
    the cursor position

Returns
-------
(line, offset): (string, integer)
    The line with the current cursor, and the character offset of the start of the line.

### Function: token_at_cursor(cell, cursor_pos)

**Description:** Get the token at a given cursor

Used for introspection.

Function calls are prioritized, so the token for the callable will be returned
if the cursor is anywhere inside the call.

Parameters
----------
cell : str
    A block of Python code
cursor_pos : int
    The location of the cursor in the block where the token should be found
