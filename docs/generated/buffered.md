## AI Summary

A file named buffered.py.


## Class: BufferedByteReceiveStream

**Description:** Wraps any bytes-based receive stream and uses a buffer to provide sophisticated
receiving capabilities in the form of a byte stream.

## Class: BufferedByteStream

**Description:** A full-duplex variant of :class:`BufferedByteReceiveStream`. All writes are passed
through to the wrapped stream as-is.

## Class: BufferedConnectable

### Function: buffer(self)

**Description:** The bytes currently in the buffer.

### Function: extra_attributes(self)

### Function: feed_data()

**Description:** Append data directly into the buffer.

Any data in the buffer will be consumed by receive operations before receiving
anything from the wrapped stream.

:param data: the data to append to the buffer (can be bytes or anything else
    that supports ``__index__()``)

### Function: __init__(self, stream)

**Description:** :param stream: the stream to be wrapped

### Function: __init__(self, connectable)

**Description:** :param connectable: the connectable to wrap
