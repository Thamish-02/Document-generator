## AI Summary

A file named _responses.py.


## Class: ResponseStream

## Class: ResponseStreamManager

## Class: AsyncResponseStream

## Class: AsyncResponseStreamManager

## Class: ResponseStreamState

### Function: __init__(self)

### Function: __next__(self)

### Function: __iter__(self)

### Function: __enter__(self)

### Function: __stream__(self)

### Function: __exit__(self, exc_type, exc, exc_tb)

### Function: close(self)

**Description:** Close the response and release the connection.

Automatically called if the response body is read to completion.

### Function: get_final_response(self)

**Description:** Waits until the stream has been read to completion and returns
the accumulated `ParsedResponse` object.

### Function: until_done(self)

**Description:** Blocks until the stream has been consumed.

### Function: __init__(self, api_request)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc, exc_tb)

### Function: __init__(self)

### Function: __init__(self, api_request)

### Function: __init__(self)

### Function: handle_event(self, event)

### Function: accumulate_event(self, event)

### Function: _create_initial_response(self, event)
