## AI Summary

A file named stapled.py.


## Class: StapledByteStream

**Description:** Combines two byte streams into a single, bidirectional byte stream.

Extra attributes will be provided from both streams, with the receive stream
providing the values in case of a conflict.

:param ByteSendStream send_stream: the sending byte stream
:param ByteReceiveStream receive_stream: the receiving byte stream

## Class: StapledObjectStream

**Description:** Combines two object streams into a single, bidirectional object stream.

Extra attributes will be provided from both streams, with the receive stream
providing the values in case of a conflict.

:param ObjectSendStream send_stream: the sending object stream
:param ObjectReceiveStream receive_stream: the receiving object stream

## Class: MultiListener

**Description:** Combines multiple listeners into one, serving connections from all of them at once.

Any MultiListeners in the given collection of listeners will have their listeners
moved into this one.

Extra attributes are provided from each listener, with each successive listener
overriding any conflicting attributes from the previous one.

:param listeners: listeners to serve
:type listeners: Sequence[Listener[T_Stream]]

### Function: extra_attributes(self)

### Function: extra_attributes(self)

### Function: __post_init__(self)

### Function: extra_attributes(self)
