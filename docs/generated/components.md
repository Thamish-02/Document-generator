## AI Summary

A file named components.py.


## Class: ComponentNotAvailable

## Class: Component

**Description:** A component managed by a debug adapter: client, launcher, or debug server.

Every component belongs to a Session, which is used for synchronization and
shared data.

Every component has its own message channel, and provides message handlers for
that channel. All handlers should be decorated with @Component.message_handler,
which ensures that Session is locked for the duration of the handler. Thus, only
one handler is running at any given time across all components, unless the lock
is released explicitly or via Session.wait_for().

Components report changes to their attributes to Session, allowing one component
to wait_for() a change caused by another component.

### Function: missing(session, type)

## Class: Capabilities

**Description:** A collection of feature flags for a component. Corresponds to JSON properties
in the DAP "initialize" request or response, other than those that identify the
party.

### Function: __init__(self, type)

### Function: __init__(self, session, stream, channel)

### Function: __str__(self)

### Function: client(self)

### Function: launcher(self)

### Function: server(self)

### Function: wait_for(self)

### Function: message_handler(f)

**Description:** Applied to a message handler to automatically lock and unlock the session
for its duration, and to validate the session state.

If the handler raises ComponentNotAvailable or JsonIOError, converts it to
Message.cant_handle().

### Function: disconnect(self)

## Class: Missing

**Description:** A dummy component that raises ComponentNotAvailable whenever some
attribute is accessed on it.

### Function: report()

### Function: __init__(self, component, message)

**Description:** Parses an "initialize" request or response and extracts the feature flags.

For every "X" in self.PROPERTIES, sets self["X"] to the corresponding value
from message.payload if it's present there, or to the default value otherwise.

### Function: __repr__(self)

### Function: require(self)

### Function: lock_and_handle(self, message)
