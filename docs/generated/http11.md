## AI Summary

A file named http11.py.


## Class: HTTPConnectionState

## Class: HTTP11Connection

## Class: HTTP11ConnectionByteStream

## Class: HTTP11UpgradeStream

### Function: __init__(self, origin, stream, keepalive_expiry)

### Function: handle_request(self, request)

### Function: _send_request_headers(self, request)

### Function: _send_request_body(self, request)

### Function: _send_event(self, event, timeout)

### Function: _receive_response_headers(self, request)

### Function: _receive_response_body(self, request)

### Function: _receive_event(self, timeout)

### Function: _response_closed(self)

### Function: close(self)

### Function: can_handle_request(self, origin)

### Function: is_available(self)

### Function: has_expired(self)

### Function: is_idle(self)

### Function: is_closed(self)

### Function: info(self)

### Function: __repr__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self, connection, request)

### Function: __iter__(self)

### Function: close(self)

### Function: __init__(self, stream, leading_data)

### Function: read(self, max_bytes, timeout)

### Function: write(self, buffer, timeout)

### Function: close(self)

### Function: start_tls(self, ssl_context, server_hostname, timeout)

### Function: get_extra_info(self, info)
