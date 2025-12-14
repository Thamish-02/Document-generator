## AI Summary

A file named decorator.py.


### Function: authorized(action, resource, message)

**Description:** A decorator for tornado.web.RequestHandler methods
that verifies whether the current user is authorized
to make the following request.

Helpful for adding an 'authorization' layer to
a REST API.

.. versionadded:: 2.0

Parameters
----------
action : str
    the type of permission or action to check.

resource: str or None
    the name of the resource the action is being authorized
    to access.

message : str or none
    a message for the unauthorized action.

### Function: allow_unauthenticated(method)

**Description:** A decorator for tornado.web.RequestHandler methods
that allows any user to make the following request.

Selectively disables the 'authentication' layer of REST API which
is active when `ServerApp.allow_unauthenticated_access = False`.

To be used exclusively on endpoints which may be considered public,
for example the login page handler.

.. versionadded:: 2.13

Parameters
----------
method : bound callable
    the endpoint method to remove authentication from.

### Function: ws_authenticated(method)

**Description:** A decorator for websockets derived from `WebSocketHandler`
that authenticates user before allowing to proceed.

Differently from tornado.web.authenticated, does not redirect
to the login page, which would be meaningless for websockets.

.. versionadded:: 2.13

Parameters
----------
method : bound callable
    the endpoint method to add authentication for.

### Function: wrapper(method)

### Function: wrapper(self)

### Function: wrapper(self)
