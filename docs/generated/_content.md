## AI Summary

A file named _content.py.


## Class: ByteStream

## Class: IteratorByteStream

## Class: AsyncIteratorByteStream

## Class: UnattachedStream

**Description:** If a request or response is serialized using pickle, then it is no longer
attached to a stream for I/O purposes. Any stream operations should result
in `httpx.StreamClosed`.

### Function: encode_content(content)

### Function: encode_urlencoded_data(data)

### Function: encode_multipart_data(data, files, boundary)

### Function: encode_text(text)

### Function: encode_html(html)

### Function: encode_json(json)

### Function: encode_request(content, data, files, json, boundary)

**Description:** Handles encoding the given `content`, `data`, `files`, and `json`,
returning a two-tuple of (<headers>, <stream>).

### Function: encode_response(content, text, html, json)

**Description:** Handles encoding the given `content`, returning a two-tuple of
(<headers>, <stream>).

### Function: __init__(self, stream)

### Function: __iter__(self)

### Function: __init__(self, stream)

### Function: __iter__(self)

### Function: __init__(self, stream)

### Function: __iter__(self)
