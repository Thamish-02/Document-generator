## AI Summary

A file named ctx.py.


## Class: _AppCtxGlobals

**Description:** A plain object. Used as a namespace for storing data during an
application context.

Creating an app context automatically creates this object, which is
made available as the :data:`g` proxy.

.. describe:: 'key' in g

    Check whether an attribute is present.

    .. versionadded:: 0.10

.. describe:: iter(g)

    Return an iterator over the attribute names.

    .. versionadded:: 0.10

### Function: after_this_request(f)

**Description:** Executes a function after this request.  This is useful to modify
response objects.  The function is passed the response object and has
to return the same or a new one.

Example::

    @app.route('/')
    def index():
        @after_this_request
        def add_header(response):
            response.headers['X-Foo'] = 'Parachute'
            return response
        return 'Hello World!'

This is more useful if a function other than the view function wants to
modify a response.  For instance think of a decorator that wants to add
some headers without converting the return value into a response object.

.. versionadded:: 0.9

### Function: copy_current_request_context(f)

**Description:** A helper function that decorates a function to retain the current
request context.  This is useful when working with greenlets.  The moment
the function is decorated a copy of the request context is created and
then pushed when the function is called.  The current session is also
included in the copied request context.

Example::

    import gevent
    from flask import copy_current_request_context

    @app.route('/')
    def index():
        @copy_current_request_context
        def do_some_work():
            # do some work here, it can access flask.request or
            # flask.session like you would otherwise in the view function.
            ...
        gevent.spawn(do_some_work)
        return 'Regular response'

.. versionadded:: 0.10

### Function: has_request_context()

**Description:** If you have code that wants to test if a request context is there or
not this function can be used.  For instance, you may want to take advantage
of request information if the request object is available, but fail
silently if it is unavailable.

::

    class User(db.Model):

        def __init__(self, username, remote_addr=None):
            self.username = username
            if remote_addr is None and has_request_context():
                remote_addr = request.remote_addr
            self.remote_addr = remote_addr

Alternatively you can also just test any of the context bound objects
(such as :class:`request` or :class:`g`) for truthness::

    class User(db.Model):

        def __init__(self, username, remote_addr=None):
            self.username = username
            if remote_addr is None and request:
                remote_addr = request.remote_addr
            self.remote_addr = remote_addr

.. versionadded:: 0.7

### Function: has_app_context()

**Description:** Works like :func:`has_request_context` but for the application
context.  You can also just do a boolean check on the
:data:`current_app` object instead.

.. versionadded:: 0.9

## Class: AppContext

**Description:** The app context contains application-specific information. An app
context is created and pushed at the beginning of each request if
one is not already active. An app context is also pushed when
running CLI commands.

## Class: RequestContext

**Description:** The request context contains per-request information. The Flask
app creates and pushes it at the beginning of the request, then pops
it at the end of the request. It will create the URL adapter and
request object for the WSGI environment provided.

Do not attempt to use this class directly, instead use
:meth:`~flask.Flask.test_request_context` and
:meth:`~flask.Flask.request_context` to create this object.

When the request context is popped, it will evaluate all the
functions registered on the application for teardown execution
(:meth:`~flask.Flask.teardown_request`).

The request context is automatically popped at the end of the
request. When using the interactive debugger, the context will be
restored so ``request`` is still accessible. Similarly, the test
client can preserve the context after the request ends. However,
teardown functions may already have closed some resources such as
database connections.

### Function: __getattr__(self, name)

### Function: __setattr__(self, name, value)

### Function: __delattr__(self, name)

### Function: get(self, name, default)

**Description:** Get an attribute by name, or a default value. Like
:meth:`dict.get`.

:param name: Name of attribute to get.
:param default: Value to return if the attribute is not present.

.. versionadded:: 0.10

### Function: pop(self, name, default)

**Description:** Get and remove an attribute by name. Like :meth:`dict.pop`.

:param name: Name of attribute to pop.
:param default: Value to return if the attribute is not present,
    instead of raising a ``KeyError``.

.. versionadded:: 0.11

### Function: setdefault(self, name, default)

**Description:** Get the value of an attribute if it is present, otherwise
set and return a default value. Like :meth:`dict.setdefault`.

:param name: Name of attribute to get.
:param default: Value to set and return if the attribute is not
    present.

.. versionadded:: 0.11

### Function: __contains__(self, item)

### Function: __iter__(self)

### Function: __repr__(self)

### Function: wrapper()

### Function: __init__(self, app)

### Function: push(self)

**Description:** Binds the app context to the current context.

### Function: pop(self, exc)

**Description:** Pops the app context.

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, tb)

### Function: __init__(self, app, environ, request, session)

### Function: copy(self)

**Description:** Creates a copy of this request context with the same request object.
This can be used to move a request context to a different greenlet.
Because the actual request object is the same this cannot be used to
move a request context to a different thread unless access to the
request object is locked.

.. versionadded:: 0.10

.. versionchanged:: 1.1
   The current session object is used instead of reloading the original
   data. This prevents `flask.session` pointing to an out-of-date object.

### Function: match_request(self)

**Description:** Can be overridden by a subclass to hook into the matching
of the request.

### Function: push(self)

### Function: pop(self, exc)

**Description:** Pops the request context and unbinds it by doing that.  This will
also trigger the execution of functions registered by the
:meth:`~flask.Flask.teardown_request` decorator.

.. versionchanged:: 0.9
   Added the `exc` argument.

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, tb)

### Function: __repr__(self)
