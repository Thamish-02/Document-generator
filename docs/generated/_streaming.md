## AI Summary

A file named _streaming.py.


## Class: Stream

**Description:** Provides the core interface to iterate over a synchronous stream response.

## Class: AsyncStream

**Description:** Provides the core interface to iterate over an asynchronous stream response.

## Class: ServerSentEvent

## Class: SSEDecoder

## Class: SSEBytesDecoder

### Function: is_stream_class_type(typ)

**Description:** TypeGuard for determining whether or not the given type is a subclass of `Stream` / `AsyncStream`

### Function: extract_stream_chunk_type(stream_cls)

**Description:** Given a type like `Stream[T]`, returns the generic type variable `T`.

This also handles the case where a concrete subclass is given, e.g.
```py
class MyStream(Stream[bytes]):
    ...

extract_stream_chunk_type(MyStream) -> bytes
```

### Function: __init__(self)

### Function: __next__(self)

### Function: __iter__(self)

### Function: _iter_events(self)

### Function: __stream__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc, exc_tb)

### Function: close(self)

**Description:** Close the response and release the connection.

Automatically called if the response body is read to completion.

### Function: __init__(self)

### Function: __init__(self)

### Function: event(self)

### Function: id(self)

### Function: retry(self)

### Function: data(self)

### Function: json(self)

### Function: __repr__(self)

### Function: __init__(self)

### Function: iter_bytes(self, iterator)

**Description:** Given an iterator that yields raw binary data, iterate over it & yield every event encountered

### Function: _iter_chunks(self, iterator)

**Description:** Given an iterator that yields raw binary data, iterate over it and yield individual SSE chunks

### Function: decode(self, line)

### Function: iter_bytes(self, iterator)

**Description:** Given an iterator that yields raw binary data, iterate over it & yield every event encountered

### Function: aiter_bytes(self, iterator)

**Description:** Given an async iterator that yields raw binary data, iterate over it & yield every event encountered
