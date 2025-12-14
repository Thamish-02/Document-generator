## AI Summary

A file named http2.py.


### Function: has_body_headers(request)

## Class: HTTPConnectionState

## Class: HTTP2Connection

## Class: HTTP2ConnectionByteStream

### Function: __init__(self, origin, stream, keepalive_expiry)

### Function: handle_request(self, request)

### Function: _send_connection_init(self, request)

**Description:** The HTTP/2 connection requires some initial setup before we can start
using individual request/response streams on it.

### Function: _send_request_headers(self, request, stream_id)

**Description:** Send the request headers to a given stream ID.

### Function: _send_request_body(self, request, stream_id)

**Description:** Iterate over the request body sending it to a given stream ID.

### Function: _send_stream_data(self, request, stream_id, data)

**Description:** Send a single chunk of data in one or more data frames.

### Function: _send_end_stream(self, request, stream_id)

**Description:** Send an empty data frame on on a given stream ID with the END_STREAM flag set.

### Function: _receive_response(self, request, stream_id)

**Description:** Return the response status code and headers for a given stream ID.

### Function: _receive_response_body(self, request, stream_id)

**Description:** Iterator that returns the bytes of the response body for a given stream ID.

### Function: _receive_stream_event(self, request, stream_id)

**Description:** Return the next available event for a given stream ID.

Will read more data from the network if required.

### Function: _receive_events(self, request, stream_id)

**Description:** Read some data from the network until we see one or more events
for a given stream ID.

### Function: _receive_remote_settings_change(self, event)

### Function: _response_closed(self, stream_id)

### Function: close(self)

### Function: _read_incoming_data(self, request)

### Function: _write_outgoing_data(self, request)

### Function: _wait_for_outgoing_flow(self, request, stream_id)

**Description:** Returns the maximum allowable outgoing flow for a given stream.

If the allowable flow is zero, then waits on the network until
WindowUpdated frames have increased the flow rate.
https://tools.ietf.org/html/rfc7540#section-6.9

### Function: can_handle_request(self, origin)

### Function: is_available(self)

### Function: has_expired(self)

### Function: is_idle(self)

### Function: is_closed(self)

### Function: info(self)

### Function: __repr__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self, connection, request, stream_id)

### Function: __iter__(self)

### Function: close(self)
