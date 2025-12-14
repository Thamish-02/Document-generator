## AI Summary

A file named kernelbase.py.


### Function: _accepts_parameters(meth, param_names)

## Class: Kernel

**Description:** The base kernel class.

### Function: _update_eventloop(self, change)

**Description:** schedule call to eventloop from IOLoop

### Function: _shell_streams_default(self)

### Function: _shell_streams_changed(self, change)

### Function: _default_ident(self)

### Function: _parent_header(self)

### Function: __init__(self)

**Description:** Initialize the kernel.

### Function: should_handle(self, stream, msg, idents)

**Description:** Check whether a shell-channel message should be handled

Allows subclasses to prevent handling of certain messages (e.g. aborted requests).

.. versionchanged:: 7
    Subclass should_handle _may_ be async.
    Base class implementation is not async.

### Function: pre_handler_hook(self)

**Description:** Hook to execute before calling message handler

### Function: post_handler_hook(self)

**Description:** Hook to execute after calling message handler

### Function: enter_eventloop(self)

**Description:** enter eventloop

### Function: start(self)

**Description:** register dispatchers for streams

### Function: record_ports(self, ports)

**Description:** Record the ports that this kernel is using.

The creator of the Kernel instance must call this methods if they
want the :meth:`connect_request` method to return the port numbers.

### Function: _publish_execute_input(self, code, parent, execution_count)

**Description:** Publish the code request on the iopub stream.

### Function: _publish_status(self, status, channel, parent)

**Description:** send status (busy/idle) on IOPub

### Function: _publish_status_and_flush(self, status, channel, stream, parent)

**Description:** send status on IOPub and flush specified stream to ensure reply is sent before handling the next reply

### Function: _publish_debug_event(self, event)

### Function: set_parent(self, ident, parent, channel)

**Description:** Set the current parent request

Side effects (IOPub messages) and replies are associated with
the request that caused them via the parent_header.

The parent identity is used to route input_request messages
on the stdin channel.

### Function: get_parent(self, channel)

**Description:** Get the parent request associated with a channel.

.. versionadded:: 6

Parameters
----------
channel : str
    the name of the channel ('shell' or 'control')

Returns
-------
message : dict
    the parent message for the most recent request on the channel.

### Function: _get_shell_context_var(self, var)

**Description:** Lookup a ContextVar, falling back on the shell context

Allows for user-launched Threads to still resolve to the shell's main context

necessary for e.g. display from threads.

### Function: send_response(self, stream, msg_or_type, content, ident, buffers, track, header, metadata, channel)

**Description:** Send a response to the message we're currently processing.

This accepts all the parameters of :meth:`jupyter_client.session.Session.send`
except ``parent``.

This relies on :meth:`set_parent` having been called for the current
message.

### Function: init_metadata(self, parent)

**Description:** Initialize metadata.

Run at the beginning of execution requests.

### Function: finish_metadata(self, parent, metadata, reply_content)

**Description:** Finish populating metadata.

Run after completing an execution request.

### Function: do_execute(self, code, silent, store_history, user_expressions, allow_stdin)

**Description:** Execute user code. Must be overridden by subclasses.

### Function: do_complete(self, code, cursor_pos)

**Description:** Override in subclasses to find completions.

### Function: do_inspect(self, code, cursor_pos, detail_level, omit_sections)

**Description:** Override in subclasses to allow introspection.

### Function: do_history(self, hist_access_type, output, raw, session, start, stop, n, pattern, unique)

**Description:** Override in subclasses to access history.

### Function: kernel_info(self)

### Function: _send_interrupt_children(self)

### Function: do_shutdown(self, restart)

**Description:** Override in subclasses to do things when the frontend shuts down the
kernel.

### Function: do_is_complete(self, code)

**Description:** Override in subclasses to find completions.

### Function: get_process_metric_value(self, process, name, attribute)

**Description:** Get the process metric value.

### Function: _topic(self, topic)

**Description:** prefixed topic for IOPub messages

### Function: _post_dummy_stop_aborting_message(self, subshell_id)

**Description:** Post a dummy message to the correct subshell that when handled will unset
the _aborting flag.

### Function: _abort_queues(self, subshell_id)

### Function: _send_abort_reply(self, stream, msg, idents)

**Description:** Send a reply to an aborted request

### Function: _no_raw_input(self)

**Description:** Raise StdinNotImplementedError if active frontend doesn't support
stdin.

### Function: getpass(self, prompt, stream)

**Description:** Forward getpass to frontends

Raises
------
StdinNotImplementedError if active frontend doesn't support stdin.

### Function: raw_input(self, prompt)

**Description:** Forward raw_input to frontends

Raises
------
StdinNotImplementedError if active frontend doesn't support stdin.

### Function: _input_request(self, prompt, ident, parent, password)

### Function: _signal_children(self, signum)

**Description:** Send a signal to all our children

Like `killpg`, but does not include the current process
(or possible parents).

### Function: _process_children(self)

**Description:** Retrieve child processes in the kernel's process group

Avoids:
- including parents and self with killpg
- including all children that may have forked-off a new group

### Function: _supports_kernel_subshells(self)

### Function: schedule_next()

**Description:** Schedule the next advance of the eventloop
