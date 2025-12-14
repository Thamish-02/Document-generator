## AI Summary

A file named _receivebuffer.py.


## Class: ReceiveBuffer

### Function: __init__(self)

### Function: __iadd__(self, byteslike)

### Function: __bool__(self)

### Function: __len__(self)

### Function: __bytes__(self)

### Function: _extract(self, count)

### Function: maybe_extract_at_most(self, count)

**Description:** Extract a fixed number of bytes from the buffer.

### Function: maybe_extract_next_line(self)

**Description:** Extract the first line, if it is completed in the buffer.

### Function: maybe_extract_lines(self)

**Description:** Extract everything up to the first blank line, and return a list of lines.

### Function: is_next_line_obviously_invalid_request_line(self)
