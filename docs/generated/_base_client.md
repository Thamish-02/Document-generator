## AI Summary

A file named _base_client.py.


## Class: PageInfo

**Description:** Stores the necessary information to build the request to retrieve the next page.

Either `url` or `params` must be set.

## Class: BasePage

**Description:** Defines the core interface for pagination.

Type Args:
    ModelT: The pydantic model that represents an item in the response.

Methods:
    has_next_page(): Check if there is another page available
    next_page_info(): Get the necessary information to make a request for the next page

## Class: BaseSyncPage

## Class: AsyncPaginator

## Class: BaseAsyncPage

## Class: BaseClient

## Class: _DefaultHttpxClient

## Class: SyncHttpxClientWrapper

## Class: SyncAPIClient

## Class: _DefaultAsyncHttpxClient

## Class: AsyncHttpxClientWrapper

## Class: AsyncAPIClient

### Function: make_request_options()

**Description:** Create a dict of type RequestOptions without keys of NotGiven values.

## Class: ForceMultipartDict

## Class: OtherPlatform

### Function: get_platform()

### Function: platform_headers(version)

## Class: OtherArch

### Function: get_python_runtime()

### Function: get_python_version()

### Function: get_architecture()

### Function: _merge_mappings(obj1, obj2)

**Description:** Merge two mappings of the same type, removing any values that are instances of `Omit`.

In cases with duplicate keys the second mapping takes precedence.

### Function: __init__(self)

### Function: __init__(self)

### Function: __init__(self)

### Function: __init__(self)

### Function: __repr__(self)

### Function: has_next_page(self)

### Function: next_page_info(self)

### Function: _get_page_items(self)

### Function: _params_from_url(self, url)

### Function: _info_to_options(self, info)

### Function: _set_private_attributes(self, client, model, options)

### Function: __iter__(self)

### Function: iter_pages(self)

### Function: get_next_page(self)

### Function: __init__(self, client, options, page_cls, model)

### Function: __await__(self)

### Function: _set_private_attributes(self, model, client, options)

### Function: __init__(self)

### Function: _enforce_trailing_slash(self, url)

### Function: _make_status_error_from_response(self, response)

### Function: _make_status_error(self, err_msg)

### Function: _build_headers(self, options)

### Function: _prepare_url(self, url)

**Description:** Merge a URL argument together with any 'base_url' on the client,
to create the URL used for the outgoing request.

### Function: _make_sse_decoder(self)

### Function: _build_request(self, options)

### Function: _serialize_multipartform(self, data)

### Function: _maybe_override_cast_to(self, cast_to, options)

### Function: _should_stream_response_body(self, request)

### Function: _process_response_data(self)

### Function: qs(self)

### Function: custom_auth(self)

### Function: auth_headers(self)

### Function: default_headers(self)

### Function: default_query(self)

### Function: _validate_headers(self, headers, custom_headers)

**Description:** Validate the given default headers and custom headers.

Does nothing by default.

### Function: user_agent(self)

### Function: base_url(self)

### Function: base_url(self, url)

### Function: platform_headers(self)

### Function: _parse_retry_after_header(self, response_headers)

**Description:** Returns a float of the number of seconds (not milliseconds) to wait after retrying, or None if unspecified.

About the Retry-After header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
See also  https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After#syntax

### Function: _calculate_retry_timeout(self, remaining_retries, options, response_headers)

### Function: _should_retry(self, response)

### Function: _idempotency_key(self)

### Function: __init__(self)

### Function: __del__(self)

### Function: __init__(self)

### Function: is_closed(self)

### Function: close(self)

**Description:** Close the underlying HTTPX client.

The client will *not* be usable after this.

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc, exc_tb)

### Function: _prepare_options(self, options)

**Description:** Hook for mutating the given options

### Function: _prepare_request(self, request)

**Description:** This method is used as a callback for mutating the `Request` object
after it has been constructed.
This is useful for cases where you want to add certain headers based off of
the request properties, e.g. `url`, `method` etc.

### Function: request(self, cast_to, options)

### Function: request(self, cast_to, options)

### Function: request(self, cast_to, options)

### Function: request(self, cast_to, options)

### Function: _sleep_for_retry(self)

### Function: _process_response(self)

### Function: _request_api_list(self, model, page, options)

### Function: get(self, path)

### Function: get(self, path)

### Function: get(self, path)

### Function: get(self, path)

### Function: post(self, path)

### Function: post(self, path)

### Function: post(self, path)

### Function: post(self, path)

### Function: patch(self, path)

### Function: put(self, path)

### Function: delete(self, path)

### Function: get_api_list(self, path)

### Function: __init__(self)

## Class: _DefaultAioHttpClient

### Function: __del__(self)

### Function: __init__(self)

### Function: is_closed(self)

### Function: _request_api_list(self, model, page, options)

### Function: get_api_list(self, path)

### Function: __bool__(self)

### Function: __init__(self, name)

### Function: __str__(self)

### Function: __init__(self, name)

### Function: __str__(self)

### Function: _parser(resp)

### Function: _parser(resp)

## Class: _DefaultAioHttpClient

### Function: __init__(self)

### Function: __init__(self)
