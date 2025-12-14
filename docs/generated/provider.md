## AI Summary

A file named provider.py.


## Class: JSONProvider

**Description:** A standard set of JSON operations for an application. Subclasses
of this can be used to customize JSON behavior or use different
JSON libraries.

To implement a provider for a specific library, subclass this base
class and implement at least :meth:`dumps` and :meth:`loads`. All
other methods have default implementations.

To use a different provider, either subclass ``Flask`` and set
:attr:`~flask.Flask.json_provider_class` to a provider class, or set
:attr:`app.json <flask.Flask.json>` to an instance of the class.

:param app: An application instance. This will be stored as a
    :class:`weakref.proxy` on the :attr:`_app` attribute.

.. versionadded:: 2.2

### Function: _default(o)

## Class: DefaultJSONProvider

**Description:** Provide JSON operations using Python's built-in :mod:`json`
library. Serializes the following additional data types:

-   :class:`datetime.datetime` and :class:`datetime.date` are
    serialized to :rfc:`822` strings. This is the same as the HTTP
    date format.
-   :class:`uuid.UUID` is serialized to a string.
-   :class:`dataclasses.dataclass` is passed to
    :func:`dataclasses.asdict`.
-   :class:`~markupsafe.Markup` (or any object with a ``__html__``
    method) will call the ``__html__`` method to get a string.

### Function: __init__(self, app)

### Function: dumps(self, obj)

**Description:** Serialize data as JSON.

:param obj: The data to serialize.
:param kwargs: May be passed to the underlying JSON library.

### Function: dump(self, obj, fp)

**Description:** Serialize data as JSON and write to a file.

:param obj: The data to serialize.
:param fp: A file opened for writing text. Should use the UTF-8
    encoding to be valid JSON.
:param kwargs: May be passed to the underlying JSON library.

### Function: loads(self, s)

**Description:** Deserialize data as JSON.

:param s: Text or UTF-8 bytes.
:param kwargs: May be passed to the underlying JSON library.

### Function: load(self, fp)

**Description:** Deserialize data as JSON read from a file.

:param fp: A file opened for reading text or UTF-8 bytes.
:param kwargs: May be passed to the underlying JSON library.

### Function: _prepare_response_obj(self, args, kwargs)

### Function: response(self)

**Description:** Serialize the given arguments as JSON, and return a
:class:`~flask.Response` object with the ``application/json``
mimetype.

The :func:`~flask.json.jsonify` function calls this method for
the current application.

Either positional or keyword arguments can be given, not both.
If no arguments are given, ``None`` is serialized.

:param args: A single value to serialize, or multiple values to
    treat as a list to serialize.
:param kwargs: Treat as a dict to serialize.

### Function: dumps(self, obj)

**Description:** Serialize data as JSON to a string.

Keyword arguments are passed to :func:`json.dumps`. Sets some
parameter defaults from the :attr:`default`,
:attr:`ensure_ascii`, and :attr:`sort_keys` attributes.

:param obj: The data to serialize.
:param kwargs: Passed to :func:`json.dumps`.

### Function: loads(self, s)

**Description:** Deserialize data as JSON from a string or bytes.

:param s: Text or UTF-8 bytes.
:param kwargs: Passed to :func:`json.loads`.

### Function: response(self)

**Description:** Serialize the given arguments as JSON, and return a
:class:`~flask.Response` object with it. The response mimetype
will be "application/json" and can be changed with
:attr:`mimetype`.

If :attr:`compact` is ``False`` or debug mode is enabled, the
output will be formatted to be easier to read.

Either positional or keyword arguments can be given, not both.
If no arguments are given, ``None`` is serialized.

:param args: A single value to serialize, or multiple values to
    treat as a list to serialize.
:param kwargs: Treat as a dict to serialize.
