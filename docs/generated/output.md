## AI Summary

A file named output.py.


## Class: CaptureOutput

**Description:** Captures output from the specified file descriptor, and tees it into another
file descriptor while generating DAP "output" events for it.

### Function: wait_for_remaining_output()

**Description:** Waits for all remaining output to be captured and propagated.

### Function: __init__(self, whose, category, fd, stream)

### Function: __del__(self)

### Function: _worker(self)

### Function: _process_chunk(self, s, final)
