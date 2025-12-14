## AI Summary

A file named _response.py.


## Class: BaseAPIResponse

## Class: APIResponse

## Class: AsyncAPIResponse

## Class: BinaryAPIResponse

**Description:** Subclass of APIResponse providing helpers for dealing with binary data.

Note: If you want to stream the response data instead of eagerly reading it
all at once then you should use `.with_streaming_response` when making
the API request, e.g. `.with_streaming_response.get_binary_response()`

## Class: AsyncBinaryAPIResponse

**Description:** Subclass of APIResponse providing helpers for dealing with binary data.

Note: If you want to stream the response data instead of eagerly reading it
all at once then you should use `.with_streaming_response` when making
the API request, e.g. `.with_streaming_response.get_binary_response()`

## Class: StreamedBinaryAPIResponse

## Class: AsyncStreamedBinaryAPIResponse

## Class: MissingStreamClassError

## Class: StreamAlreadyConsumed

**Description:** Attempted to read or stream content, but the content has already
been streamed.

This can happen if you use a method like `.iter_lines()` and then attempt
to read th entire response body afterwards, e.g.

```py
response = await client.post(...)
async for line in response.iter_lines():
    ...  # do something with `line`

content = await response.read()
# ^ error
```

If you want this behaviour you'll need to either manually accumulate the response
content or call `await response.read()` before iterating over the stream.

## Class: ResponseContextManager

**Description:** Context manager for ensuring that a request is not made
until it is entered and that the response will always be closed
when the context manager exits

## Class: AsyncResponseContextManager

**Description:** Context manager for ensuring that a request is not made
until it is entered and that the response will always be closed
when the context manager exits

### Function: to_streamed_response_wrapper(func)

**Description:** Higher order function that takes one of our bound API methods and wraps it
to support streaming and returning the raw `APIResponse` object directly.

### Function: async_to_streamed_response_wrapper(func)

**Description:** Higher order function that takes one of our bound API methods and wraps it
to support streaming and returning the raw `APIResponse` object directly.

### Function: to_custom_streamed_response_wrapper(func, response_cls)

**Description:** Higher order function that takes one of our bound API methods and an `APIResponse` class
and wraps the method to support streaming and returning the given response class directly.

Note: the given `response_cls` *must* be concrete, e.g. `class BinaryAPIResponse(APIResponse[bytes])`

### Function: async_to_custom_streamed_response_wrapper(func, response_cls)

**Description:** Higher order function that takes one of our bound API methods and an `APIResponse` class
and wraps the method to support streaming and returning the given response class directly.

Note: the given `response_cls` *must* be concrete, e.g. `class BinaryAPIResponse(APIResponse[bytes])`

### Function: to_raw_response_wrapper(func)

**Description:** Higher order function that takes one of our bound API methods and wraps it
to support returning the raw `APIResponse` object directly.

### Function: async_to_raw_response_wrapper(func)

**Description:** Higher order function that takes one of our bound API methods and wraps it
to support returning the raw `APIResponse` object directly.

### Function: to_custom_raw_response_wrapper(func, response_cls)

**Description:** Higher order function that takes one of our bound API methods and an `APIResponse` class
and wraps the method to support returning the given response class directly.

Note: the given `response_cls` *must* be concrete, e.g. `class BinaryAPIResponse(APIResponse[bytes])`

### Function: async_to_custom_raw_response_wrapper(func, response_cls)

**Description:** Higher order function that takes one of our bound API methods and an `APIResponse` class
and wraps the method to support returning the given response class directly.

Note: the given `response_cls` *must* be concrete, e.g. `class BinaryAPIResponse(APIResponse[bytes])`

### Function: extract_response_type(typ)

**Description:** Given a type like `APIResponse[T]`, returns the generic type variable `T`.

This also handles the case where a concrete subclass is given, e.g.
```py
class MyResponse(APIResponse[bytes]):
    ...

extract_response_type(MyResponse) -> bytes
```

### Function: __init__(self)

### Function: headers(self)

### Function: http_request(self)

**Description:** Returns the httpx Request instance associated with the current response.

### Function: status_code(self)

### Function: url(self)

**Description:** Returns the URL for which the request was made.

### Function: method(self)

### Function: http_version(self)

### Function: elapsed(self)

**Description:** The time taken for the complete request/response cycle to complete.

### Function: is_closed(self)

**Description:** Whether or not the response body has been closed.

If this is False then there is response data that has not been read yet.
You must either fully consume the response body or call `.close()`
before discarding the response to prevent resource leaks.

### Function: __repr__(self)

### Function: _parse(self)

### Function: request_id(self)

### Function: parse(self)

### Function: parse(self)

### Function: parse(self)

**Description:** Returns the rich python representation of this response's data.

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

### Function: read(self)

**Description:** Read and return the binary response content.

### Function: text(self)

**Description:** Read and decode the response content into a string.

### Function: json(self)

**Description:** Read and decode the JSON response content.

### Function: close(self)

**Description:** Close the response and release the connection.

Automatically called if the response body is read to completion.

### Function: iter_bytes(self, chunk_size)

**Description:** A byte-iterator over the decoded response content.

This automatically handles gzip, deflate and brotli encoded responses.

### Function: iter_text(self, chunk_size)

**Description:** A str-iterator over the decoded response content
that handles both gzip, deflate, etc but also detects the content's
string encoding.

### Function: iter_lines(self)

**Description:** Like `iter_text()` but will only yield chunks for each line

### Function: request_id(self)

### Function: write_to_file(self, file)

**Description:** Write the output to the given file.

Accepts a filename or any path-like object, e.g. pathlib.Path

Note: if you want to stream the data to the file instead of writing
all at once then you should use `.with_streaming_response` when making
the API request, e.g. `.with_streaming_response.get_binary_response()`

### Function: stream_to_file(self, file)

**Description:** Streams the output to the given file.

Accepts a filename or any path-like object, e.g. pathlib.Path

### Function: __init__(self)

### Function: __init__(self)

### Function: __init__(self, request_func)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc, exc_tb)

### Function: __init__(self, api_request)

### Function: wrapped()

### Function: wrapped()

### Function: wrapped()

### Function: wrapped()

### Function: wrapped()

### Function: wrapped()

### Function: wrapped()
