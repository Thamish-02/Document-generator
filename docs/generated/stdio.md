## AI Summary

A file named stdio.py.


## Class: LspStdIoBase

**Description:** Non-blocking, queued base for communicating with stdio Language Servers

## Class: LspStdIoReader

**Description:** Language Server stdio Reader

Because non-blocking (but still synchronous) IO is used, rudimentary
exponential backoff is used.

## Class: LspStdIoWriter

**Description:** Language Server stdio Writer

### Function: __repr__(self)

### Function: __init__(self)

### Function: close(self)

### Function: _default_max_wait(self)

### Function: wake(self)

**Description:** Reset the wait time

### Function: _readline(self)

**Description:** Read a line (or immediately return None)

### Function: _write_one(self, message)
