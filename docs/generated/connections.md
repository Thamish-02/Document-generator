## AI Summary

A file named connections.py.


## Class: GatewayWebSocketConnection

**Description:** Web socket connection that proxies to a kernel/enterprise gateway.

### Function: _connection_done(self, fut)

**Description:** Handle a finished connection.

### Function: disconnect(self)

**Description:** Handle a disconnect.

### Function: handle_outgoing_message(self, incoming_msg)

**Description:** Send message to the notebook client.

### Function: handle_incoming_message(self, message)

**Description:** Send message to gateway server.

### Function: _write_message(self, message)

**Description:** Send message to gateway server.

### Function: _get_message_summary(message)

**Description:** Get a summary of a message.
