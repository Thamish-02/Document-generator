## AI Summary

A file named wsgi.py.


### Function: _skip_leading_empty_chunks(body)

## Class: WSGIByteStream

## Class: WSGITransport

**Description:** A custom transport that handles sending requests directly to an WSGI app.
The simplest way to use this functionality is to use the `app` argument.

```
client = httpx.Client(app=app)
```

Alternatively, you can setup the transport instance explicitly.
This allows you to include any additional configuration arguments specific
to the WSGITransport class:

```
transport = httpx.WSGITransport(
    app=app,
    script_name="/submount",
    remote_addr="1.2.3.4"
)
client = httpx.Client(transport=transport)
```

Arguments:

* `app` - The WSGI application.
* `raise_app_exceptions` - Boolean indicating if exceptions in the application
   should be raised. Default to `True`. Can be set to `False` for use cases
   such as testing the content of a client 500 response.
* `script_name` - The root path on which the WSGI application should be mounted.
* `remote_addr` - A string indicating the client IP of incoming requests.
```

### Function: __init__(self, result)

### Function: __iter__(self)

### Function: close(self)

### Function: __init__(self, app, raise_app_exceptions, script_name, remote_addr, wsgi_errors)

### Function: handle_request(self, request)

### Function: start_response(status, response_headers, exc_info)
