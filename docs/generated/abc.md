## AI Summary

A file named abc.py.


## Class: KernelWebsocketConnectionABC

**Description:** This class defines a minimal interface that should
be used to bridge the connection between Jupyter
Server's websocket API and a kernel's ZMQ socket
interface.

### Function: handle_incoming_message(self, incoming_msg)

**Description:** Broker the incoming websocket message to the appropriate ZMQ channel.

### Function: handle_outgoing_message(self, stream, outgoing_msg)

**Description:** Broker outgoing ZMQ messages to the kernel websocket.
