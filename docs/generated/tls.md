## AI Summary

A file named tls.py.


## Class: TLSAttribute

**Description:** Contains Transport Layer Security related attributes.

## Class: TLSStream

**Description:** A stream wrapper that encrypts all sent data and decrypts received data.

This class has no public initializer; use :meth:`wrap` instead.
All extra attributes from :class:`~TLSAttribute` are supported.

:var AnyByteStream transport_stream: the wrapped stream

## Class: TLSListener

**Description:** A convenience listener that wraps another listener and auto-negotiates a TLS session
on every accepted connection.

If the TLS handshake times out or raises an exception,
:meth:`handle_handshake_error` is called to do whatever post-mortem processing is
deemed necessary.

Supports only the :attr:`~TLSAttribute.standard_compatible` extra attribute.

:param Listener listener: the listener to wrap
:param ssl_context: the SSL context object
:param standard_compatible: a flag passed through to :meth:`TLSStream.wrap`
:param handshake_timeout: time limit for the TLS handshake
    (passed to :func:`~anyio.fail_after`)

## Class: TLSConnectable

**Description:** Wraps another connectable and does TLS negotiation after a successful connection.

:param connectable: the connectable to wrap
:param hostname: host name of the server (if host name checking is desired)
:param ssl_context: the SSLContext object to use (if not provided, a secure default
    will be created)
:param standard_compatible: if ``False``, skip the closing handshake when closing
    the connection, and don't raise an exception if the server does the same

### Function: extra_attributes(self)

### Function: extra_attributes(self)

### Function: __init__(self, connectable)
