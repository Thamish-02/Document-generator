## AI Summary

A file named parentpoller.py.


## Class: ParentPollerUnix

**Description:** A Unix-specific daemon thread that terminates the program immediately
when the parent process no longer exists.

## Class: ParentPollerWindows

**Description:** A Windows-specific daemon thread that listens for a special event that
signals an interrupt and, optionally, terminates the program immediately
when the parent process no longer exists.

### Function: __init__(self, parent_pid)

**Description:** Initialize the poller.

Parameters
----------
parent_handle : int, optional
    If provided, the program will terminate immediately when
    process parent is no longer this original parent.

### Function: run(self)

**Description:** Run the poller.

### Function: __init__(self, interrupt_handle, parent_handle)

**Description:** Create the poller. At least one of the optional parameters must be
provided.

Parameters
----------
interrupt_handle : HANDLE (int), optional
    If provided, the program will generate a Ctrl+C event when this
    handle is signaled.
parent_handle : HANDLE (int), optional
    If provided, the program will terminate immediately when this
    handle is signaled.

### Function: run(self)

**Description:** Run the poll loop. This method never returns.
