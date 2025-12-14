## AI Summary

A file named zmqhistory.py.


## Class: ZMQHistoryManager

**Description:** History accessor and manager for ZMQ-based kernels

### Function: __init__(self, client)

**Description:** Class to load the command-line history from a ZMQ-based kernel,
and access the history.

Parameters
----------

client: `IPython.kernel.KernelClient`
  The kernel client in order to request the history.

### Function: _load_history(self, raw, output, hist_access_type)

**Description:** Load the history over ZMQ from the kernel. Wraps the history
messaging with loop to wait to get history results.

### Function: get_tail(self, n, raw, output, include_latest)

### Function: search(self, pattern, raw, search_raw, output, n, unique)

### Function: get_range(self, session, start, stop, raw, output)

### Function: get_range_by_str(self, rangestr, raw, output)

### Function: end_session(self)

**Description:** Nothing to do for ZMQ-based histories.

### Function: reset(self, new_session)

**Description:** Nothing to do for ZMQ-based histories.
