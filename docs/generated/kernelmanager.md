## AI Summary

A file named kernelmanager.py.


## Class: MappingKernelManager

**Description:** A KernelManager that handles
- File mapping
- HTTP error handling
- Kernel message filtering

## Class: AsyncMappingKernelManager

**Description:** An asynchronous mapping kernel manager.

### Function: emit_kernel_action_event(success_msg)

**Description:** Decorate kernel action methods to
begin emitting jupyter kernel action events.

Parameters
----------
success_msg: str
    A formattable string that's passed to the message field of
    the emitted event when the action succeeds. You can include
    the kernel_id, kernel_name, or action in the message using
    a formatted string argument,
    e.g. "{kernel_id} succeeded to {action}."

error_msg: str
    A formattable string that's passed to the message field of
    the emitted event when the action fails. You can include
    the kernel_id, kernel_name, or action in the message using
    a formatted string argument,
    e.g. "{kernel_id} failed to {action}."

## Class: ServerKernelManager

**Description:** A server-specific kernel manager.

### Function: _default_kernel_manager_class(self)

### Function: _default_root_dir(self)

### Function: _update_root_dir(self, proposal)

**Description:** Do a bit of validation of the root dir.

### Function: _default_kernel_buffers(self)

### Function: __init__(self)

**Description:** Initialize a kernel manager.

### Function: _handle_kernel_died(self, kernel_id)

**Description:** notice that a kernel died

### Function: cwd_for_path(self, path)

**Description:** Turn API path into absolute OS path.

### Function: ports_changed(self, kernel_id)

**Description:** Used by ZMQChannelsHandler to determine how to coordinate nudge and replays.

Ports are captured when starting a kernel (via MappingKernelManager).  Ports
are considered changed (following restarts) if the referenced KernelManager
is using a set of ports different from those captured at startup.  If changes
are detected, the captured set is updated and a value of True is returned.

NOTE: Use is exclusive to ZMQChannelsHandler because this object is a singleton
instance while ZMQChannelsHandler instances are per WebSocket connection that
can vary per kernel lifetime.

### Function: _get_changed_ports(self, kernel_id)

**Description:** Internal method to test if a kernel's ports have changed and, if so, return their values.

This method does NOT update the captured ports for the kernel as that can only be done
by ZMQChannelsHandler, but instead returns the new list of ports if they are different
than those captured at startup.  This enables the ability to conditionally restart
activity monitoring immediately following a kernel's restart (if ports have changed).

### Function: start_buffering(self, kernel_id, session_key, channels)

**Description:** Start buffering messages for a kernel

Parameters
----------
kernel_id : str
    The id of the kernel to stop buffering.
session_key : str
    The session_key, if any, that should get the buffer.
    If the session_key matches the current buffered session_key,
    the buffer will be returned.
channels : dict({'channel': ZMQStream})
    The zmq channels whose messages should be buffered.

### Function: get_buffer(self, kernel_id, session_key)

**Description:** Get the buffer for a given kernel

Parameters
----------
kernel_id : str
    The id of the kernel to stop buffering.
session_key : str, optional
    The session_key, if any, that should get the buffer.
    If the session_key matches the current buffered session_key,
    the buffer will be returned.

### Function: stop_buffering(self, kernel_id)

**Description:** Stop buffering kernel messages

Parameters
----------
kernel_id : str
    The id of the kernel to stop buffering.

### Function: notify_connect(self, kernel_id)

**Description:** Notice a new connection to a kernel

### Function: notify_disconnect(self, kernel_id)

**Description:** Notice a disconnection from a kernel

### Function: kernel_model(self, kernel_id)

**Description:** Return a JSON-safe dict representing a kernel

For use in representing kernels in the JSON APIs.

### Function: list_kernels(self)

**Description:** Returns a list of kernel_id's of kernels running.

### Function: _check_kernel_id(self, kernel_id)

**Description:** Check a that a kernel_id exists and raise 404 if not.

### Function: track_message_type(self, message_type)

### Function: start_watching_activity(self, kernel_id)

**Description:** Start watching IOPub messages on a kernel for activity.

- update last_activity on every message
- record execution_state from status messages

### Function: stop_watching_activity(self, kernel_id)

**Description:** Stop watching IOPub messages on a kernel for activity.

### Function: initialize_culler(self)

**Description:** Start idle culler if 'cull_idle_timeout' is greater than zero.

Regardless of that value, set flag that we've been here.

### Function: _default_kernel_manager_class(self)

### Function: _validate_kernel_manager_class(self, proposal)

**Description:** A validator for the kernel manager class.

### Function: __init__(self)

**Description:** Initialize an async mapping kernel manager.

### Function: wrap_method(method)

### Function: core_event_schema_paths(self)

### Function: _default_event_logger(self)

**Description:** Initialize the logger and ensure all required events are present.

### Function: emit(self, schema_id, data)

**Description:** Emit an event from the kernel manager.

### Function: buffer_msg(channel, msg_parts)

### Function: finish()

**Description:** Common cleanup when restart finishes/fails for any reason.

### Function: on_reply(msg)

### Function: on_timeout()

### Function: on_restart_failed()

### Function: record_activity(msg_list)

**Description:** Record an IOPub message arriving from a kernel
