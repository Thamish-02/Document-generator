## AI Summary

A file named _headers.py.


## Class: Headers

**Description:** A list-like interface that allows iterating over headers as byte-pairs
of (lowercased-name, value).

Internally we actually store the representation as three-tuples,
including both the raw original casing, in order to preserve casing
over-the-wire, and the lowercased name, for case-insensitive comparisions.

r = Request(
    method="GET",
    target="/",
    headers=[("Host", "example.org"), ("Connection", "keep-alive")],
    http_version="1.1",
)
assert r.headers == [
    (b"host", b"example.org"),
    (b"connection", b"keep-alive")
]
assert r.headers.raw_items() == [
    (b"Host", b"example.org"),
    (b"Connection", b"keep-alive")
]

### Function: normalize_and_validate(headers, _parsed)

### Function: normalize_and_validate(headers, _parsed)

### Function: normalize_and_validate(headers, _parsed)

### Function: normalize_and_validate(headers, _parsed)

### Function: get_comma_header(headers, name)

### Function: set_comma_header(headers, name, new_values)

### Function: has_expect_100_continue(request)

### Function: __init__(self, full_items)

### Function: __bool__(self)

### Function: __eq__(self, other)

### Function: __len__(self)

### Function: __repr__(self)

### Function: __getitem__(self, idx)

### Function: raw_items(self)
