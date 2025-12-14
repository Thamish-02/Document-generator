## AI Summary

A file named channels.py.


### Function: _ensure_future(f)

**Description:** Wrap a concurrent future as an asyncio future if there is a running loop.

## Class: ZMQChannelsWebsocketConnection

**Description:** A Jupyter Server Websocket Connection

### Function: write_message(self)

**Description:** Alias to the websocket handler's write_message method.

### Function: _default_kernel_info_future(self)

**Description:** The default kernel info future.

### Function: _default_close_future(self)

**Description:** The default close future.

### Function: subprotocol(self)

**Description:** The sub protocol.

### Function: create_stream(self)

**Description:** Create a stream.

### Function: nudge(self)

**Description:** Nudge the zmq connections with kernel_info_requests
Returns a Future that will resolve when we have received
a shell or control reply and at least one iopub message,
ensuring that zmq subscriptions are established,
sockets are fully connected, and kernel is responsive.
Keeps retrying kernel_info_request until these are both received.

### Function: connect(self)

**Description:** Handle a connection.

### Function: close(self)

**Description:** Close the connection.

### Function: disconnect(self)

**Description:** Handle a disconnect.

### Function: handle_incoming_message(self, incoming_msg)

**Description:** Handle incoming messages from Websocket to ZMQ Sockets.

### Function: handle_outgoing_message(self, stream, outgoing_msg)

**Description:** Handle the outgoing messages from ZMQ sockets to Websocket.

### Function: get_part(self, field, value, msg_list)

**Description:** Get a part of a message.

### Function: _reserialize_reply(self, msg_or_list, channel)

**Description:** Reserialize a reply message using JSON.

msg_or_list can be an already-deserialized msg dict or the zmq buffer list.
If it is the zmq list, it will be deserialized with self.session.

This takes the msg list from the ZMQ socket and serializes the result for the websocket.
This method should be used by self._on_zmq_reply to build messages that can
be sent back to the browser.

### Function: _on_zmq_reply(self, stream, msg_list)

**Description:** Handle a zmq reply.

### Function: request_kernel_info(self)

**Description:** send a request for kernel_info

### Function: _handle_kernel_info_reply(self, msg)

**Description:** process the kernel_info_reply

enabling msg spec adaptation, if necessary

### Function: _finish_kernel_info(self, info)

**Description:** Finish handling kernel_info reply

Set up protocol adaptation, if needed,
and signal that connection can continue.

### Function: write_stderr(self, error_message, parent_header)

**Description:** Write a message to stderr.

### Function: _limit_rate(self, channel, msg, msg_list)

**Description:** Limit the message rate on a channel.

### Function: _send_status_message(self, status)

**Description:** Send a status message.

### Function: on_kernel_restarted(self)

**Description:** Handle a kernel restart.

### Function: on_restart_failed(self)

**Description:** Handle a kernel restart failure.

### Function: _on_error(self, channel, msg, msg_list)

**Description:** Handle an error message.

### Function: finish(_)

**Description:** Ensure all futures are resolved
which in turn triggers cleanup

### Function: cleanup(_)

**Description:** Common cleanup

### Function: on_shell_reply(msg)

**Description:** Handle nudge shell replies.

### Function: on_control_reply(msg)

**Description:** Handle nudge control replies.

### Function: on_iopub(msg)

**Description:** Handle nudge iopub replies.

### Function: nudge(count)

**Description:** Nudge the kernel.

### Function: give_up()

**Description:** Don't wait forever for the kernel to reply

### Function: subscribe(value)

### Function: replay(value)
