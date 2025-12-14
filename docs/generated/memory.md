## AI Summary

A file named memory.py.


## Class: MemoryObjectStreamStatistics

## Class: MemoryObjectItemReceiver

## Class: MemoryObjectStreamState

## Class: MemoryObjectReceiveStream

## Class: MemoryObjectSendStream

### Function: __repr__(self)

### Function: statistics(self)

### Function: __post_init__(self)

### Function: receive_nowait(self)

**Description:** Receive the next item if it can be done without waiting.

:return: the received item
:raises ~anyio.ClosedResourceError: if this send stream has been closed
:raises ~anyio.EndOfStream: if the buffer is empty and this stream has been
    closed from the sending end
:raises ~anyio.WouldBlock: if there are no items in the buffer and no tasks
    waiting to send

### Function: clone(self)

**Description:** Create a clone of this receive stream.

Each clone can be closed separately. Only when all clones have been closed will
the receiving end of the memory stream be considered closed by the sending ends.

:return: the cloned stream

### Function: close(self)

**Description:** Close the stream.

This works the exact same way as :meth:`aclose`, but is provided as a special
case for the benefit of synchronous callbacks.

### Function: statistics(self)

**Description:** Return statistics about the current state of this stream.

.. versionadded:: 3.0

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: __del__(self)

### Function: __post_init__(self)

### Function: send_nowait(self, item)

**Description:** Send an item immediately if it can be done without waiting.

:param item: the item to send
:raises ~anyio.ClosedResourceError: if this send stream has been closed
:raises ~anyio.BrokenResourceError: if the stream has been closed from the
    receiving end
:raises ~anyio.WouldBlock: if the buffer is full and there are no tasks waiting
    to receive

### Function: clone(self)

**Description:** Create a clone of this send stream.

Each clone can be closed separately. Only when all clones have been closed will
the sending end of the memory stream be considered closed by the receiving ends.

:return: the cloned stream

### Function: close(self)

**Description:** Close the stream.

This works the exact same way as :meth:`aclose`, but is provided as a special
case for the benefit of synchronous callbacks.

### Function: statistics(self)

**Description:** Return statistics about the current state of this stream.

.. versionadded:: 3.0

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: __del__(self)
