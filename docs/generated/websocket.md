## AI Summary

A file named websocket.py.


## Class: KernelWebsocketHandler

**Description:** The kernels websocket should connect

### Function: kernel_websocket_connection_class(self)

**Description:** The kernel websocket connection class.

### Function: set_default_headers(self)

**Description:** Undo the set_default_headers in JupyterHandler

which doesn't make sense for websockets

### Function: get_compression_options(self)

**Description:** Get the socket connection options.

### Function: on_message(self, ws_message)

**Description:** Get a kernel message from the websocket and turn it into a ZMQ message.

### Function: on_close(self)

**Description:** Handle a socket closure.

### Function: select_subprotocol(self, subprotocols)

**Description:** Select the sub protocol for the socket.
