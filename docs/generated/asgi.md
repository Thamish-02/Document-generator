## AI Summary

A file named asgi.py.


### Function: is_running_trio()

### Function: create_event()

## Class: ASGIResponseStream

## Class: ASGITransport

**Description:** A custom AsyncTransport that handles sending requests directly to an ASGI app.

```python
transport = httpx.ASGITransport(
    app=app,
    root_path="/submount",
    client=("1.2.3.4", 123)
)
client = httpx.AsyncClient(transport=transport)
```

Arguments:

* `app` - The ASGI application.
* `raise_app_exceptions` - Boolean indicating if exceptions in the application
   should be raised. Default to `True`. Can be set to `False` for use cases
   such as testing the content of a client 500 response.
* `root_path` - The root path on which the ASGI application should be mounted.
* `client` - A two-tuple indicating the client IP and port of incoming requests.
```

### Function: __init__(self, body)

### Function: __init__(self, app, raise_app_exceptions, root_path, client)
