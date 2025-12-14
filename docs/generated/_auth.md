## AI Summary

A file named _auth.py.


## Class: Auth

**Description:** Base class for all authentication schemes.

To implement a custom authentication scheme, subclass `Auth` and override
the `.auth_flow()` method.

If the authentication scheme does I/O such as disk access or network calls, or uses
synchronization primitives such as locks, you should override `.sync_auth_flow()`
and/or `.async_auth_flow()` instead of `.auth_flow()` to provide specialized
implementations that will be used by `Client` and `AsyncClient` respectively.

## Class: FunctionAuth

**Description:** Allows the 'auth' argument to be passed as a simple callable function,
that takes the request, and returns a new, modified request.

## Class: BasicAuth

**Description:** Allows the 'auth' argument to be passed as a (username, password) pair,
and uses HTTP Basic authentication.

## Class: NetRCAuth

**Description:** Use a 'netrc' file to lookup basic auth credentials based on the url host.

## Class: DigestAuth

## Class: _DigestAuthChallenge

### Function: auth_flow(self, request)

**Description:** Execute the authentication flow.

To dispatch a request, `yield` it:

```
yield request
```

The client will `.send()` the response back into the flow generator. You can
access it like so:

```
response = yield request
```

A `return` (or reaching the end of the generator) will result in the
client returning the last response obtained from the server.

You can dispatch as many requests as is necessary.

### Function: sync_auth_flow(self, request)

**Description:** Execute the authentication flow synchronously.

By default, this defers to `.auth_flow()`. You should override this method
when the authentication scheme does I/O and/or uses concurrency primitives.

### Function: __init__(self, func)

### Function: auth_flow(self, request)

### Function: __init__(self, username, password)

### Function: auth_flow(self, request)

### Function: _build_auth_header(self, username, password)

### Function: __init__(self, file)

### Function: auth_flow(self, request)

### Function: _build_auth_header(self, username, password)

### Function: __init__(self, username, password)

### Function: auth_flow(self, request)

### Function: _parse_challenge(self, request, response, auth_header)

**Description:** Returns a challenge from a Digest WWW-Authenticate header.
These take the form of:
`Digest realm="realm@host.com",qop="auth,auth-int",nonce="abc",opaque="xyz"`

### Function: _build_auth_header(self, request, challenge)

### Function: _get_client_nonce(self, nonce_count, nonce)

### Function: _get_header_value(self, header_fields)

### Function: _resolve_qop(self, qop, request)

### Function: digest(data)
