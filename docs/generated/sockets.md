## AI Summary

A file named sockets.py.


### Function: can_bind_ipv4_localhost()

**Description:** Check if we can bind to IPv4 localhost.

### Function: can_bind_ipv6_localhost()

**Description:** Check if we can bind to IPv6 localhost.

### Function: get_default_localhost()

**Description:** Get the default localhost address.
Defaults to IPv4 '127.0.0.1', but falls back to IPv6 '::1' if IPv4 is unavailable.

### Function: get_address(sock)

**Description:** Gets the socket address host and port.

### Function: create_server(host, port, backlog, timeout)

**Description:** Return a local server socket listening on the given port.

### Function: create_client(ipv6)

**Description:** Return a client socket that may be connected to a remote address.

### Function: _new_sock(ipv6)

### Function: shut_down(sock, how)

**Description:** Shut down the given socket.

### Function: close_socket(sock)

**Description:** Shutdown and close the socket.

### Function: serve(name, handler, host, port, backlog, timeout)

**Description:** Accepts TCP connections on the specified host and port, and invokes the
provided handler function for every new connection.

Returns the created server socket.

### Function: accept_worker()
