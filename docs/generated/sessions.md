## AI Summary

A file named sessions.py.


## Class: SessionMixin

**Description:** Expands a basic dictionary with session attributes.

## Class: SecureCookieSession

**Description:** Base class for sessions based on signed cookies.

This session backend will set the :attr:`modified` and
:attr:`accessed` attributes. It cannot reliably track whether a
session is new (vs. empty), so :attr:`new` remains hard coded to
``False``.

## Class: NullSession

**Description:** Class used to generate nicer error messages if sessions are not
available.  Will still allow read-only access to the empty session
but fail on setting.

## Class: SessionInterface

**Description:** The basic interface you have to implement in order to replace the
default session interface which uses werkzeug's securecookie
implementation.  The only methods you have to implement are
:meth:`open_session` and :meth:`save_session`, the others have
useful defaults which you don't need to change.

The session object returned by the :meth:`open_session` method has to
provide a dictionary like interface plus the properties and methods
from the :class:`SessionMixin`.  We recommend just subclassing a dict
and adding that mixin::

    class Session(dict, SessionMixin):
        pass

If :meth:`open_session` returns ``None`` Flask will call into
:meth:`make_null_session` to create a session that acts as replacement
if the session support cannot work because some requirement is not
fulfilled.  The default :class:`NullSession` class that is created
will complain that the secret key was not set.

To replace the session interface on an application all you have to do
is to assign :attr:`flask.Flask.session_interface`::

    app = Flask(__name__)
    app.session_interface = MySessionInterface()

Multiple requests with the same session may be sent and handled
concurrently. When implementing a new session interface, consider
whether reads or writes to the backing store must be synchronized.
There is no guarantee on the order in which the session for each
request is opened or saved, it will occur in the order that requests
begin and end processing.

.. versionadded:: 0.8

### Function: _lazy_sha1(string)

**Description:** Don't access ``hashlib.sha1`` until runtime. FIPS builds may not include
SHA-1, in which case the import and use as a default would fail before the
developer can configure something else.

## Class: SecureCookieSessionInterface

**Description:** The default session interface that stores sessions in signed cookies
through the :mod:`itsdangerous` module.

### Function: permanent(self)

**Description:** This reflects the ``'_permanent'`` key in the dict.

### Function: permanent(self, value)

### Function: __init__(self, initial)

### Function: __getitem__(self, key)

### Function: get(self, key, default)

### Function: setdefault(self, key, default)

### Function: _fail(self)

### Function: make_null_session(self, app)

**Description:** Creates a null session which acts as a replacement object if the
real session support could not be loaded due to a configuration
error.  This mainly aids the user experience because the job of the
null session is to still support lookup without complaining but
modifications are answered with a helpful error message of what
failed.

This creates an instance of :attr:`null_session_class` by default.

### Function: is_null_session(self, obj)

**Description:** Checks if a given object is a null session.  Null sessions are
not asked to be saved.

This checks if the object is an instance of :attr:`null_session_class`
by default.

### Function: get_cookie_name(self, app)

**Description:** The name of the session cookie. Uses``app.config["SESSION_COOKIE_NAME"]``.

### Function: get_cookie_domain(self, app)

**Description:** The value of the ``Domain`` parameter on the session cookie. If not set,
browsers will only send the cookie to the exact domain it was set from.
Otherwise, they will send it to any subdomain of the given value as well.

Uses the :data:`SESSION_COOKIE_DOMAIN` config.

.. versionchanged:: 2.3
    Not set by default, does not fall back to ``SERVER_NAME``.

### Function: get_cookie_path(self, app)

**Description:** Returns the path for which the cookie should be valid.  The
default implementation uses the value from the ``SESSION_COOKIE_PATH``
config var if it's set, and falls back to ``APPLICATION_ROOT`` or
uses ``/`` if it's ``None``.

### Function: get_cookie_httponly(self, app)

**Description:** Returns True if the session cookie should be httponly.  This
currently just returns the value of the ``SESSION_COOKIE_HTTPONLY``
config var.

### Function: get_cookie_secure(self, app)

**Description:** Returns True if the cookie should be secure.  This currently
just returns the value of the ``SESSION_COOKIE_SECURE`` setting.

### Function: get_cookie_samesite(self, app)

**Description:** Return ``'Strict'`` or ``'Lax'`` if the cookie should use the
``SameSite`` attribute. This currently just returns the value of
the :data:`SESSION_COOKIE_SAMESITE` setting.

### Function: get_cookie_partitioned(self, app)

**Description:** Returns True if the cookie should be partitioned. By default, uses
the value of :data:`SESSION_COOKIE_PARTITIONED`.

.. versionadded:: 3.1

### Function: get_expiration_time(self, app, session)

**Description:** A helper method that returns an expiration date for the session
or ``None`` if the session is linked to the browser session.  The
default implementation returns now + the permanent session
lifetime configured on the application.

### Function: should_set_cookie(self, app, session)

**Description:** Used by session backends to determine if a ``Set-Cookie`` header
should be set for this session cookie for this response. If the session
has been modified, the cookie is set. If the session is permanent and
the ``SESSION_REFRESH_EACH_REQUEST`` config is true, the cookie is
always set.

This check is usually skipped if the session was deleted.

.. versionadded:: 0.11

### Function: open_session(self, app, request)

**Description:** This is called at the beginning of each request, after
pushing the request context, before matching the URL.

This must return an object which implements a dictionary-like
interface as well as the :class:`SessionMixin` interface.

This will return ``None`` to indicate that loading failed in
some way that is not immediately an error. The request
context will fall back to using :meth:`make_null_session`
in this case.

### Function: save_session(self, app, session, response)

**Description:** This is called at the end of each request, after generating
a response, before removing the request context. It is skipped
if :meth:`is_null_session` returns ``True``.

### Function: get_signing_serializer(self, app)

### Function: open_session(self, app, request)

### Function: save_session(self, app, session, response)

### Function: on_update(self)
