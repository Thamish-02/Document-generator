## AI Summary

A file named interfaces.py.


## Class: RequestInterface

## Class: ConnectionInterface

### Function: request(self, method, url)

### Function: stream(self, method, url)

### Function: handle_request(self, request)

### Function: close(self)

### Function: info(self)

### Function: can_handle_request(self, origin)

### Function: is_available(self)

**Description:** Return `True` if the connection is currently able to accept an
outgoing request.

An HTTP/1.1 connection will only be available if it is currently idle.

An HTTP/2 connection will be available so long as the stream ID space is
not yet exhausted, and the connection is not in an error state.

While the connection is being established we may not yet know if it is going
to result in an HTTP/1.1 or HTTP/2 connection. The connection should be
treated as being available, but might ultimately raise `NewConnectionRequired`
required exceptions if multiple requests are attempted over a connection
that ends up being established as HTTP/1.1.

### Function: has_expired(self)

**Description:** Return `True` if the connection is in a state where it should be closed.

This either means that the connection is idle and it has passed the
expiry time on its keep-alive, or that server has sent an EOF.

### Function: is_idle(self)

**Description:** Return `True` if the connection is currently idle.

### Function: is_closed(self)

**Description:** Return `True` if the connection has been closed.

Used when a response is closed to determine if the connection may be
returned to the connection pool or not.
