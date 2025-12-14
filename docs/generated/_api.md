## AI Summary

A file named _api.py.


### Function: request(method, url)

**Description:** Sends an HTTP request.

**Parameters:**

* **method** - HTTP method for the new `Request` object: `GET`, `OPTIONS`,
`HEAD`, `POST`, `PUT`, `PATCH`, or `DELETE`.
* **url** - URL for the new `Request` object.
* **params** - *(optional)* Query parameters to include in the URL, as a
string, dictionary, or sequence of two-tuples.
* **content** - *(optional)* Binary content to include in the body of the
request, as bytes or a byte iterator.
* **data** - *(optional)* Form data to include in the body of the request,
as a dictionary.
* **files** - *(optional)* A dictionary of upload files to include in the
body of the request.
* **json** - *(optional)* A JSON serializable object to include in the body
of the request.
* **headers** - *(optional)* Dictionary of HTTP headers to include in the
request.
* **cookies** - *(optional)* Dictionary of Cookie items to include in the
request.
* **auth** - *(optional)* An authentication class to use when sending the
request.
* **proxy** - *(optional)* A proxy URL where all the traffic should be routed.
* **timeout** - *(optional)* The timeout configuration to use when sending
the request.
* **follow_redirects** - *(optional)* Enables or disables HTTP redirects.
* **verify** - *(optional)* Either `True` to use an SSL context with the
default CA bundle, `False` to disable verification, or an instance of
`ssl.SSLContext` to use a custom context.
* **trust_env** - *(optional)* Enables or disables usage of environment
variables for configuration.

**Returns:** `Response`

Usage:

```
>>> import httpx
>>> response = httpx.request('GET', 'https://httpbin.org/get')
>>> response
<Response [200 OK]>
```

### Function: stream(method, url)

**Description:** Alternative to `httpx.request()` that streams the response body
instead of loading it into memory at once.

**Parameters**: See `httpx.request`.

See also: [Streaming Responses][0]

[0]: /quickstart#streaming-responses

### Function: get(url)

**Description:** Sends a `GET` request.

**Parameters**: See `httpx.request`.

Note that the `data`, `files`, `json` and `content` parameters are not available
on this function, as `GET` requests should not include a request body.

### Function: options(url)

**Description:** Sends an `OPTIONS` request.

**Parameters**: See `httpx.request`.

Note that the `data`, `files`, `json` and `content` parameters are not available
on this function, as `OPTIONS` requests should not include a request body.

### Function: head(url)

**Description:** Sends a `HEAD` request.

**Parameters**: See `httpx.request`.

Note that the `data`, `files`, `json` and `content` parameters are not available
on this function, as `HEAD` requests should not include a request body.

### Function: post(url)

**Description:** Sends a `POST` request.

**Parameters**: See `httpx.request`.

### Function: put(url)

**Description:** Sends a `PUT` request.

**Parameters**: See `httpx.request`.

### Function: patch(url)

**Description:** Sends a `PATCH` request.

**Parameters**: See `httpx.request`.

### Function: delete(url)

**Description:** Sends a `DELETE` request.

**Parameters**: See `httpx.request`.

Note that the `data`, `files`, `json` and `content` parameters are not available
on this function, as `DELETE` requests should not include a request body.
