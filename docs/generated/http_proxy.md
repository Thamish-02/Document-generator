## AI Summary

A file named http_proxy.py.


### Function: merge_headers(default_headers, override_headers)

**Description:** Append default_headers and override_headers, de-duplicating if a key exists
in both cases.

## Class: HTTPProxy

**Description:** A connection pool that sends requests via an HTTP proxy.

## Class: ForwardHTTPConnection

## Class: TunnelHTTPConnection

### Function: __init__(self, proxy_url, proxy_auth, proxy_headers, ssl_context, proxy_ssl_context, max_connections, max_keepalive_connections, keepalive_expiry, http1, http2, retries, local_address, uds, network_backend, socket_options)

**Description:** A connection pool for making HTTP requests.

Parameters:
    proxy_url: The URL to use when connecting to the proxy server.
        For example `"http://127.0.0.1:8080/"`.
    proxy_auth: Any proxy authentication as a two-tuple of
        (username, password). May be either bytes or ascii-only str.
    proxy_headers: Any HTTP headers to use for the proxy requests.
        For example `{"Proxy-Authorization": "Basic <username>:<password>"}`.
    ssl_context: An SSL context to use for verifying connections.
        If not specified, the default `httpcore.default_ssl_context()`
        will be used.
    proxy_ssl_context: The same as `ssl_context`, but for a proxy server rather than a remote origin.
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
    retries: The maximum number of retries when trying to establish
        a connection.
    local_address: Local address to connect from. Can also be used to
        connect using a particular address family. Using
        `local_address="0.0.0.0"` will connect using an `AF_INET` address
        (IPv4), while using `local_address="::"` will connect using an
        `AF_INET6` address (IPv6).
    uds: Path to a Unix Domain Socket to use instead of TCP sockets.
    network_backend: A backend instance to use for handling network I/O.

### Function: create_connection(self, origin)

### Function: __init__(self, proxy_origin, remote_origin, proxy_headers, keepalive_expiry, network_backend, socket_options, proxy_ssl_context)

### Function: handle_request(self, request)

### Function: can_handle_request(self, origin)

### Function: close(self)

### Function: info(self)

### Function: is_available(self)

### Function: has_expired(self)

### Function: is_idle(self)

### Function: is_closed(self)

### Function: __repr__(self)

### Function: __init__(self, proxy_origin, remote_origin, ssl_context, proxy_ssl_context, proxy_headers, keepalive_expiry, http1, http2, network_backend, socket_options)

### Function: handle_request(self, request)

### Function: can_handle_request(self, origin)

### Function: close(self)

### Function: info(self)

### Function: is_available(self)

### Function: has_expired(self)

### Function: is_idle(self)

### Function: is_closed(self)

### Function: __repr__(self)
