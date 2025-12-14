## AI Summary

A file named socket_pair.py.


## Class: SocketPair

**Description:** Pair of ZMQ inproc sockets for one-direction communication between 2 threads.

One of the threads is always the shell_channel_thread, the other may be the control
thread, main thread or a subshell thread.

.. versionadded:: 7

### Function: __init__(self, context, name)

**Description:** Initialize the inproc socker pair.

### Function: close(self)

**Description:** Close the inproc socker pair.

### Function: on_recv(self, io_loop, on_recv_callback, copy)

**Description:** Set the callback used when a message is received on the to stream.

### Function: _address(self, name)

**Description:** Return the address used for this inproc socket pair.
