## AI Summary

A file named _sockets.py.


### Function: getnameinfo(sockaddr, flags)

**Description:** Look up the host name of an IP address.

:param sockaddr: socket address (e.g. (ipaddress, port) for IPv4)
:param flags: flags to pass to upstream ``getnameinfo()``
:return: a tuple of (host name, service name)

.. seealso:: :func:`socket.getnameinfo`

### Function: wait_socket_readable(sock)

**Description:** .. deprecated:: 4.7.0
   Use :func:`wait_readable` instead.

Wait until the given socket has data to be read.

.. warning:: Only use this on raw sockets that have not been wrapped by any higher
    level constructs like socket streams!

:param sock: a socket object
:raises ~anyio.ClosedResourceError: if the socket was closed while waiting for the
    socket to become readable
:raises ~anyio.BusyResourceError: if another task is already waiting for the socket
    to become readable

### Function: wait_socket_writable(sock)

**Description:** .. deprecated:: 4.7.0
   Use :func:`wait_writable` instead.

Wait until the given socket can be written to.

This does **NOT** work on Windows when using the asyncio backend with a proactor
event loop (default on py3.8+).

.. warning:: Only use this on raw sockets that have not been wrapped by any higher
    level constructs like socket streams!

:param sock: a socket object
:raises ~anyio.ClosedResourceError: if the socket was closed while waiting for the
    socket to become writable
:raises ~anyio.BusyResourceError: if another task is already waiting for the socket
    to become writable

### Function: wait_readable(obj)

**Description:** Wait until the given object has data to be read.

On Unix systems, ``obj`` must either be an integer file descriptor, or else an
object with a ``.fileno()`` method which returns an integer file descriptor. Any
kind of file descriptor can be passed, though the exact semantics will depend on
your kernel. For example, this probably won't do anything useful for on-disk files.

On Windows systems, ``obj`` must either be an integer ``SOCKET`` handle, or else an
object with a ``.fileno()`` method which returns an integer ``SOCKET`` handle. File
descriptors aren't supported, and neither are handles that refer to anything besides
a ``SOCKET``.

On backends where this functionality is not natively provided (asyncio
``ProactorEventLoop`` on Windows), it is provided using a separate selector thread
which is set to shut down when the interpreter shuts down.

.. warning:: Don't use this on raw sockets that have been wrapped by any higher
    level constructs like socket streams!

:param obj: an object with a ``.fileno()`` method or an integer handle
:raises ~anyio.ClosedResourceError: if the object was closed while waiting for the
    object to become readable
:raises ~anyio.BusyResourceError: if another task is already waiting for the object
    to become readable

### Function: wait_writable(obj)

**Description:** Wait until the given object can be written to.

:param obj: an object with a ``.fileno()`` method or an integer handle
:raises ~anyio.ClosedResourceError: if the object was closed while waiting for the
    object to become writable
:raises ~anyio.BusyResourceError: if another task is already waiting for the object
    to become writable

.. seealso:: See the documentation of :func:`wait_readable` for the definition of
   ``obj`` and notes on backend compatibility.

.. warning:: Don't use this on raw sockets that have been wrapped by any higher
    level constructs like socket streams!

### Function: notify_closing(obj)

**Description:** Call this before closing a file descriptor (on Unix) or socket (on
Windows). This will cause any `wait_readable` or `wait_writable`
calls on the given object to immediately wake up and raise
`~anyio.ClosedResourceError`.

This doesn't actually close the object – you still have to do that
yourself afterwards. Also, you want to be careful to make sure no
new tasks start waiting on the object in between when you call this
and when it's actually closed. So to close something properly, you
usually want to do these steps in order:

1. Explicitly mark the object as closed, so that any new attempts
   to use it will abort before they start.
2. Call `notify_closing` to wake up any already-existing users.
3. Actually close the object.

It's also possible to do them in a different order if that's more
convenient, *but only if* you make sure not to have any checkpoints in
between the steps. This way they all happen in a single atomic
step, so other tasks won't be able to tell what order they happened
in anyway.

:param obj: an object with a ``.fileno()`` method or an integer handle

### Function: convert_ipv6_sockaddr(sockaddr)

**Description:** Convert a 4-tuple IPv6 socket address to a 2-tuple (address, port) format.

If the scope ID is nonzero, it is added to the address, separated with ``%``.
Otherwise the flow id and scope id are simply cut off from the tuple.
Any other kinds of socket addresses are returned as-is.

:param sockaddr: the result of :meth:`~socket.socket.getsockname`
:return: the converted socket address

## Class: TCPConnectable

**Description:** Connects to a TCP server at the given host and port.

:param host: host name or IP address of the server
:param port: TCP port number of the server

## Class: UNIXConnectable

**Description:** Connects to a UNIX domain socket at the given path.

:param path: the file system path of the socket

### Function: as_connectable()

**Description:** Return a byte stream connectable from the given object.

If a bytestream connectable is given, it is returned unchanged.
If a tuple of (host, port) is given, a TCP connectable is returned.
If a string or bytes path is given, a UNIX connectable is returned.

If ``tls=True``, the connectable will be wrapped in a
:class:`~.streams.tls.TLSConnectable`.

:param remote: a connectable, a tuple of (host, port) or a path to a UNIX socket
:param tls: if ``True``, wrap the plaintext connectable in a
    :class:`~.streams.tls.TLSConnectable`, using the provided TLS settings)
:param ssl_context: if ``tls=True``, the SSLContext object to use  (if not provided,
    a secure default will be created)
:param tls_hostname: if ``tls=True``, host name of the server to use for checking
    the server certificate (defaults to the host portion of the address for TCP
    connectables)
:param tls_standard_compatible: if ``False`` and ``tls=True``, makes the TLS stream
    skip the closing handshake when closing the connection, so it won't raise an
    exception if the server does the same

### Function: setup_raw_socket(fam, bind_addr)

### Function: __post_init__(self)
