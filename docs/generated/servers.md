## AI Summary

A file named servers.py.


## Class: Connection

**Description:** A debug server that is connected to the adapter.

Servers that are not participating in a debug session are managed directly by the
corresponding Connection instance.

Servers that are participating in a debug session are managed by that sessions's
Server component instance, but Connection object remains, and takes over again
once the session ends.

## Class: Server

**Description:** Handles the debug server side of a debug session.

### Function: serve(host, port)

### Function: is_serving()

### Function: stop_serving()

### Function: connections()

### Function: wait_for_connection(session, predicate, timeout)

**Description:** Waits until there is a server matching the specified predicate connected to
this adapter, and returns the corresponding Connection.

If there is more than one server connection already available, returns the oldest
one.

### Function: wait_until_disconnected()

**Description:** Blocks until all debug servers disconnect from the adapter.

If there are no server connections, waits until at least one is established first,
before waiting for it to disconnect.

### Function: dont_wait_for_first_connection()

**Description:** Unblocks any pending wait_until_disconnected() call that is waiting on the
first server to connect.

### Function: inject(pid, debugpy_args, on_output)

### Function: __init__(self, sock)

### Function: __str__(self)

### Function: authenticate(self)

### Function: request(self, request)

### Function: event(self, event)

### Function: terminated_event(self, event)

### Function: disconnect(self)

### Function: attach_to_session(self, session)

**Description:** Attaches this server to the specified Session as a Server component.

Raises ValueError if the server already belongs to some session.

## Class: Capabilities

### Function: __init__(self, session, connection)

### Function: pid(self)

**Description:** Process ID of the debuggee process, as reported by the server.

### Function: ppid(self)

**Description:** Parent process ID of the debuggee process, as reported by the server.

### Function: initialize(self, request)

### Function: request(self, request)

### Function: event(self, event)

### Function: initialized_event(self, event)

### Function: process_event(self, event)

### Function: continued_event(self, event)

### Function: exited_event(self, event)

### Function: terminated_event(self, event)

### Function: detach_from_session(self)

### Function: disconnect(self)

### Function: wait_for_timeout()

### Function: capture(stream)

### Function: info_on_timeout()
