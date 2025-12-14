## AI Summary

A file named sync.py.


## Class: TLSinTLSStream

**Description:** Because the standard `SSLContext.wrap_socket` method does
not work for `SSLSocket` objects, we need this class
to implement TLS stream using an underlying `SSLObject`
instance in order to support TLS on top of TLS.

## Class: SyncStream

## Class: SyncBackend

### Function: __init__(self, sock, ssl_context, server_hostname, timeout)

### Function: _perform_io(self, func)

### Function: read(self, max_bytes, timeout)

### Function: write(self, buffer, timeout)

### Function: close(self)

### Function: start_tls(self, ssl_context, server_hostname, timeout)

### Function: get_extra_info(self, info)

### Function: __init__(self, sock)

### Function: read(self, max_bytes, timeout)

### Function: write(self, buffer, timeout)

### Function: close(self)

### Function: start_tls(self, ssl_context, server_hostname, timeout)

### Function: get_extra_info(self, info)

### Function: connect_tcp(self, host, port, timeout, local_address, socket_options)

### Function: connect_unix_socket(self, path, timeout, socket_options)
