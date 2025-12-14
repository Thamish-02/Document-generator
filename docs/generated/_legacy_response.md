## AI Summary

A file named _legacy_response.py.


## Class: LegacyAPIResponse

**Description:** This is a legacy class as it will be replaced by `APIResponse`
and `AsyncAPIResponse` in the `_response.py` file in the next major
release.

For the sync client this will mostly be the same with the exception
of `content` & `text` will be methods instead of properties. In the
async client, all methods will be async.

A migration script will be provided & the migration in general should
be smooth.

## Class: MissingStreamClassError

### Function: to_raw_response_wrapper(func)

**Description:** Higher order function that takes one of our bound API methods and wraps it
to support returning the raw `APIResponse` object directly.

### Function: async_to_raw_response_wrapper(func)

**Description:** Higher order function that takes one of our bound API methods and wraps it
to support returning the raw `APIResponse` object directly.

## Class: HttpxBinaryResponseContent

### Function: __init__(self)

### Function: request_id(self)

### Function: parse(self)

### Function: parse(self)

### Function: parse(self)

**Description:** Returns the rich python representation of this response's data.

NOTE: For the async client: this will become a coroutine in the next major version.

For lower-level control, see `.read()`, `.json()`, `.iter_bytes()`.

You can customise the type that the response is parsed into through
the `to` argument, e.g.

```py
from openai import BaseModel


class MyModel(BaseModel):
    foo: str


obj = response.parse(to=MyModel)
print(obj.foo)
```

We support parsing:
  - `BaseModel`
  - `dict`
  - `list`
  - `Union`
  - `str`
  - `int`
  - `float`
  - `httpx.Response`

### Function: headers(self)

### Function: http_request(self)

### Function: status_code(self)

### Function: url(self)

### Function: method(self)

### Function: content(self)

**Description:** Return the binary response content.

NOTE: this will be removed in favour of `.read()` in the
next major version.

### Function: text(self)

**Description:** Return the decoded response content.

NOTE: this will be turned into a method in the next major version.

### Function: http_version(self)

### Function: is_closed(self)

### Function: elapsed(self)

**Description:** The time taken for the complete request/response cycle to complete.

### Function: _parse(self)

### Function: __repr__(self)

### Function: __init__(self)

### Function: wrapped()

### Function: __init__(self, response)

### Function: content(self)

### Function: text(self)

### Function: encoding(self)

### Function: charset_encoding(self)

### Function: json(self)

### Function: read(self)

### Function: iter_bytes(self, chunk_size)

### Function: iter_text(self, chunk_size)

### Function: iter_lines(self)

### Function: iter_raw(self, chunk_size)

### Function: write_to_file(self, file)

**Description:** Write the output to the given file.

Accepts a filename or any path-like object, e.g. pathlib.Path

Note: if you want to stream the data to the file instead of writing
all at once then you should use `.with_streaming_response` when making
the API request, e.g. `client.with_streaming_response.foo().stream_to_file('my_filename.txt')`

### Function: stream_to_file(self, file)

### Function: close(self)
