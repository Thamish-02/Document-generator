## AI Summary

A file named connection_pool.py.


## Class: PoolRequest

## Class: ConnectionPool

**Description:** A connection pool for making HTTP requests.

## Class: PoolByteStream

### Function: __init__(self, request)

### Function: assign_to_connection(self, connection)

### Function: clear_connection(self)

### Function: wait_for_connection(self, timeout)

### Function: is_queued(self)

### Function: __init__(self, ssl_context, proxy, max_connections, max_keepalive_connections, keepalive_expiry, http1, http2, retries, local_address, uds, network_backend, socket_options)

**Description:** A connection pool for making HTTP requests.

Parameters:
    ssl_context: An SSL context to use for verifying connections.
        If not specified, the default `httpcore.default_ssl_context()`
        will be used.
    max_connections: The maximum number of concurrent HTTP connections that
        the pool should allow. Any attempt to send a request on a pool that
        would exceed this amount will block until a connection is available.
    max_keepalive_connections: The maximum number of idle HTTP connections
        that will be maintained in the pool.
    keepalive_expiry: The duration in seconds that an idle HTTP connection
        may be maintained for before being expired from the pool.
    http1: A boolean indicating if HTTP/1.1 requests should be supported
        by the connection pool. Defaults to True.
    http2: A boolean indicating if HTTP/2 requests should be supported by
        the connection pool. Defaults to False.
    retries: The maximum number of retries when trying to establish a
        connection.
    local_address: Local address to connect from. Can also be used to connect
        using a particular address family. Using `local_address="0.0.0.0"`
        will connect using an `AF_INET` address (IPv4), while using
        `local_address="::"` will connect using an `AF_INET6` address (IPv6).
    uds: Path to a Unix Domain Socket to use instead of TCP sockets.
    network_backend: A backend instance to use for handling network I/O.
    socket_options: Socket options that have to be included
     in the TCP socket when the connection was established.

### Function: create_connection(self, origin)

### Function: connections(self)

**Description:** Return a list of the connections currently in the pool.

For example:

```python
>>> pool.connections
[
    <HTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 6]>,
    <HTTPConnection ['https://example.com:443', HTTP/1.1, IDLE, Request Count: 9]> ,
    <HTTPConnection ['http://example.com:80', HTTP/1.1, IDLE, Request Count: 1]>,
]
```

### Function: handle_request(self, request)

**Description:** Send an HTTP request, and return an HTTP response.

This is the core implementation that is called into by `.request()` or `.stream()`.

### Function: _assign_requests_to_connections(self)

**Description:** Manage the state of the connection pool, assigning incoming
requests to connections as available.

Called whenever a new request is added or removed from the pool.

Any closing connections are returned, allowing the I/O for closing
those connections to be handled seperately.

### Function: _close_connections(self, closing)

### Function: close(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __repr__(self)

### Function: __init__(self, stream, pool_request, pool)

### Function: __iter__(self)

### Function: close(self)
