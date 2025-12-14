## AI Summary

A file named _util.py.


## Class: ProtocolError

**Description:** Exception indicating a violation of the HTTP/1.1 protocol.

This as an abstract base class, with two concrete base classes:
:exc:`LocalProtocolError`, which indicates that you tried to do something
that HTTP/1.1 says is illegal, and :exc:`RemoteProtocolError`, which
indicates that the remote peer tried to do something that HTTP/1.1 says is
illegal. See :ref:`error-handling` for details.

In addition to the normal :exc:`Exception` features, it has one attribute:

.. attribute:: error_status_hint

   This gives a suggestion as to what status code a server might use if
   this error occurred as part of a request.

   For a :exc:`RemoteProtocolError`, this is useful as a suggestion for
   how you might want to respond to a misbehaving peer, if you're
   implementing a server.

   For a :exc:`LocalProtocolError`, this can be taken as a suggestion for
   how your peer might have responded to *you* if h11 had allowed you to
   continue.

   The default is 400 Bad Request, a generic catch-all for protocol
   violations.

## Class: LocalProtocolError

## Class: RemoteProtocolError

### Function: validate(regex, data, msg)

## Class: Sentinel

### Function: bytesify(s)

### Function: __init__(self, msg, error_status_hint)

### Function: _reraise_as_remote_protocol_error(self)

### Function: __new__(cls, name, bases, namespace)

### Function: __repr__(self)
