## AI Summary

A file named base_comm.py.


## Class: BaseComm

**Description:** Class for communicating between a Frontend and a Kernel

Must be subclassed with a publish_msg method implementation which
sends comm messages through the iopub channel.

## Class: CommManager

**Description:** Default CommManager singleton implementation for Comms in the Kernel

### Function: __init__(self, target_name, data, metadata, buffers, comm_id, primary, target_module, topic, _open_data, _close_data)

### Function: publish_msg(self, msg_type, data, metadata, buffers)

### Function: __del__(self)

**Description:** trigger close on gc

### Function: open(self, data, metadata, buffers)

**Description:** Open the frontend-side version of this comm

### Function: close(self, data, metadata, buffers, deleting)

**Description:** Close the frontend-side version of this comm

### Function: send(self, data, metadata, buffers)

**Description:** Send a message to the frontend-side version of this comm

### Function: on_close(self, callback)

**Description:** Register a callback for comm_close

Will be called with the `data` of the close message.

Call `on_close(None)` to disable an existing callback.

### Function: on_msg(self, callback)

**Description:** Register a callback for comm_msg

Will be called with the `data` of any comm_msg messages.

Call `on_msg(None)` to disable an existing callback.

### Function: handle_close(self, msg)

**Description:** Handle a comm_close message

### Function: handle_msg(self, msg)

**Description:** Handle a comm_msg message

### Function: __init__(self)

### Function: register_target(self, target_name, f)

**Description:** Register a callable f for a given target name

f will be called with two arguments when a comm_open message is received with `target`:

- the Comm instance
- the `comm_open` message itself.

f can be a Python callable or an import string for one.

### Function: unregister_target(self, target_name, f)

**Description:** Unregister a callable registered with register_target

### Function: register_comm(self, comm)

**Description:** Register a new comm

### Function: unregister_comm(self, comm)

**Description:** Unregister a comm, and close its counterpart

### Function: get_comm(self, comm_id)

**Description:** Get a comm with a particular id

Returns the comm if found, otherwise None.

This will not raise an error,
it will log messages if the comm cannot be found.

### Function: comm_open(self, stream, ident, msg)

**Description:** Handler for comm_open messages

### Function: comm_msg(self, stream, ident, msg)

**Description:** Handler for comm_msg messages

### Function: comm_close(self, stream, ident, msg)

**Description:** Handler for comm_close messages
