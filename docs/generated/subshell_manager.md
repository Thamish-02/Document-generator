## AI Summary

A file named subshell_manager.py.


## Class: SubshellManager

**Description:** A manager of subshells.

Controls the lifetimes of subshell threads and their associated ZMQ sockets and
streams. Runs mostly in the shell channel thread.

Care needed with threadsafe access here.  All write access to the cache occurs in
the shell channel thread so there is only ever one write access at any one time.
Reading of cache information can be performed by other threads, so all reads are
protected by a lock so that they are atomic.

Sending reply messages via the shell_socket is wrapped by another lock to protect
against multiple subshells attempting to send at the same time.

.. versionadded:: 7

### Function: __init__(self, context, shell_channel_io_loop, shell_socket)

**Description:** Initialize the subshell manager.

### Function: close(self)

**Description:** Stop all subshells and close all resources.

### Function: get_shell_channel_to_subshell_pair(self, subshell_id)

**Description:** Return the inproc socket pair used to send messages from the shell channel
to a particular subshell or main shell.

### Function: get_subshell_to_shell_channel_socket(self, subshell_id)

**Description:** Return the socket used by a particular subshell or main shell to send
messages to the shell channel.

### Function: get_shell_channel_to_subshell_socket(self, subshell_id)

**Description:** Return the socket used by the shell channel to send messages to a particular
subshell or main shell.

### Function: get_subshell_aborting(self, subshell_id)

**Description:** Get the boolean aborting flag of the specified subshell.

### Function: get_subshell_asyncio_lock(self, subshell_id)

**Description:** Return the asyncio lock belonging to the specified subshell.

### Function: list_subshell(self)

**Description:** Return list of current subshell ids.

Can be called by any subshell using %subshell magic.

### Function: set_on_recv_callback(self, on_recv_callback)

**Description:** Set the callback used by the main shell and all subshells to receive
messages sent from the shell channel thread.

### Function: set_subshell_aborting(self, subshell_id, aborting)

**Description:** Set the aborting flag of the specified subshell.

### Function: subshell_id_from_thread_id(self, thread_id)

**Description:** Return subshell_id of the specified thread_id.

Raises RuntimeError if thread_id is not the main shell or a subshell.

Only used by %subshell magic so does not have to be fast/cached.

### Function: _create_subshell(self)

**Description:** Create and start a new subshell thread.

### Function: _delete_subshell(self, subshell_id)

**Description:** Delete subshell identified by subshell_id.

Raises key error if subshell_id not in cache.

### Function: _process_control_request(self, request)

**Description:** Process a control request message received on the control inproc
socket and return the reply.  Runs in the shell channel thread.

### Function: _send_on_shell_channel(self, msg)

### Function: _stop_subshell(self, subshell_thread)

**Description:** Stop a subshell thread and close all of its resources.
