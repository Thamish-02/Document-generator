## AI Summary

A file named _urls.py.


## Class: URL

**Description:** url = httpx.URL("HTTPS://jo%40email.com:a%20secret@müller.de:1234/pa%20th?search=ab#anchorlink")

assert url.scheme == "https"
assert url.username == "jo@email.com"
assert url.password == "a secret"
assert url.userinfo == b"jo%40email.com:a%20secret"
assert url.host == "müller.de"
assert url.raw_host == b"xn--mller-kva.de"
assert url.port == 1234
assert url.netloc == b"xn--mller-kva.de:1234"
assert url.path == "/pa th"
assert url.query == b"?search=ab"
assert url.raw_path == b"/pa%20th?search=ab"
assert url.fragment == "anchorlink"

The components of a URL are broken down like this:

   https://jo%40email.com:a%20secret@müller.de:1234/pa%20th?search=ab#anchorlink
[scheme]   [  username  ] [password] [ host ][port][ path ] [ query ] [fragment]
           [       userinfo        ] [   netloc   ][    raw_path    ]

Note that:

* `url.scheme` is normalized to always be lowercased.

* `url.host` is normalized to always be lowercased. Internationalized domain
  names are represented in unicode, without IDNA encoding applied. For instance:

  url = httpx.URL("http://中国.icom.museum")
  assert url.host == "中国.icom.museum"
  url = httpx.URL("http://xn--fiqs8s.icom.museum")
  assert url.host == "中国.icom.museum"

* `url.raw_host` is normalized to always be lowercased, and is IDNA encoded.

  url = httpx.URL("http://中国.icom.museum")
  assert url.raw_host == b"xn--fiqs8s.icom.museum"
  url = httpx.URL("http://xn--fiqs8s.icom.museum")
  assert url.raw_host == b"xn--fiqs8s.icom.museum"

* `url.port` is either None or an integer. URLs that include the default port for
  "http", "https", "ws", "wss", and "ftp" schemes have their port
  normalized to `None`.

  assert httpx.URL("http://example.com") == httpx.URL("http://example.com:80")
  assert httpx.URL("http://example.com").port is None
  assert httpx.URL("http://example.com:80").port is None

* `url.userinfo` is raw bytes, without URL escaping. Usually you'll want to work
  with `url.username` and `url.password` instead, which handle the URL escaping.

* `url.raw_path` is raw bytes of both the path and query, without URL escaping.
  This portion is used as the target when constructing HTTP requests. Usually you'll
  want to work with `url.path` instead.

* `url.query` is raw bytes, without URL escaping. A URL query string portion can
  only be properly URL escaped when decoding the parameter names and values
  themselves.

## Class: QueryParams

**Description:** URL query parameters, as a multi-dict.

### Function: __init__(self, url)

### Function: scheme(self)

**Description:** The URL scheme, such as "http", "https".
Always normalised to lowercase.

### Function: raw_scheme(self)

**Description:** The raw bytes representation of the URL scheme, such as b"http", b"https".
Always normalised to lowercase.

### Function: userinfo(self)

**Description:** The URL userinfo as a raw bytestring.
For example: b"jo%40email.com:a%20secret".

### Function: username(self)

**Description:** The URL username as a string, with URL decoding applied.
For example: "jo@email.com"

### Function: password(self)

**Description:** The URL password as a string, with URL decoding applied.
For example: "a secret"

### Function: host(self)

**Description:** The URL host as a string.
Always normalized to lowercase, with IDNA hosts decoded into unicode.

Examples:

url = httpx.URL("http://www.EXAMPLE.org")
assert url.host == "www.example.org"

url = httpx.URL("http://中国.icom.museum")
assert url.host == "中国.icom.museum"

url = httpx.URL("http://xn--fiqs8s.icom.museum")
assert url.host == "中国.icom.museum"

url = httpx.URL("https://[::ffff:192.168.0.1]")
assert url.host == "::ffff:192.168.0.1"

### Function: raw_host(self)

**Description:** The raw bytes representation of the URL host.
Always normalized to lowercase, and IDNA encoded.

Examples:

url = httpx.URL("http://www.EXAMPLE.org")
assert url.raw_host == b"www.example.org"

url = httpx.URL("http://中国.icom.museum")
assert url.raw_host == b"xn--fiqs8s.icom.museum"

url = httpx.URL("http://xn--fiqs8s.icom.museum")
assert url.raw_host == b"xn--fiqs8s.icom.museum"

url = httpx.URL("https://[::ffff:192.168.0.1]")
assert url.raw_host == b"::ffff:192.168.0.1"

### Function: port(self)

**Description:** The URL port as an integer.

Note that the URL class performs port normalization as per the WHATWG spec.
Default ports for "http", "https", "ws", "wss", and "ftp" schemes are always
treated as `None`.

For example:

assert httpx.URL("http://www.example.com") == httpx.URL("http://www.example.com:80")
assert httpx.URL("http://www.example.com:80").port is None

### Function: netloc(self)

**Description:** Either `<host>` or `<host>:<port>` as bytes.
Always normalized to lowercase, and IDNA encoded.

This property may be used for generating the value of a request
"Host" header.

### Function: path(self)

**Description:** The URL path as a string. Excluding the query string, and URL decoded.

For example:

url = httpx.URL("https://example.com/pa%20th")
assert url.path == "/pa th"

### Function: query(self)

**Description:** The URL query string, as raw bytes, excluding the leading b"?".

This is necessarily a bytewise interface, because we cannot
perform URL decoding of this representation until we've parsed
the keys and values into a QueryParams instance.

For example:

url = httpx.URL("https://example.com/?filter=some%20search%20terms")
assert url.query == b"filter=some%20search%20terms"

### Function: params(self)

**Description:** The URL query parameters, neatly parsed and packaged into an immutable
multidict representation.

### Function: raw_path(self)

**Description:** The complete URL path and query string as raw bytes.
Used as the target when constructing HTTP requests.

For example:

GET /users?search=some%20text HTTP/1.1
Host: www.example.org
Connection: close

### Function: fragment(self)

**Description:** The URL fragments, as used in HTML anchors.
As a string, without the leading '#'.

### Function: is_absolute_url(self)

**Description:** Return `True` for absolute URLs such as 'http://example.com/path',
and `False` for relative URLs such as '/path'.

### Function: is_relative_url(self)

**Description:** Return `False` for absolute URLs such as 'http://example.com/path',
and `True` for relative URLs such as '/path'.

### Function: copy_with(self)

**Description:** Copy this URL, returning a new URL with some components altered.
Accepts the same set of parameters as the components that are made
available via properties on the `URL` class.

For example:

url = httpx.URL("https://www.example.com").copy_with(
    username="jo@gmail.com", password="a secret"
)
assert url == "https://jo%40email.com:a%20secret@www.example.com"

### Function: copy_set_param(self, key, value)

### Function: copy_add_param(self, key, value)

### Function: copy_remove_param(self, key)

### Function: copy_merge_params(self, params)

### Function: join(self, url)

**Description:** Return an absolute URL, using this URL as the base.

Eg.

url = httpx.URL("https://www.example.com/test")
url = url.join("/new/path")
assert url == "https://www.example.com/new/path"

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: __str__(self)

### Function: __repr__(self)

### Function: raw(self)

### Function: __init__(self)

### Function: keys(self)

**Description:** Return all the keys in the query params.

Usage:

q = httpx.QueryParams("a=123&a=456&b=789")
assert list(q.keys()) == ["a", "b"]

### Function: values(self)

**Description:** Return all the values in the query params. If a key occurs more than once
only the first item for that key is returned.

Usage:

q = httpx.QueryParams("a=123&a=456&b=789")
assert list(q.values()) == ["123", "789"]

### Function: items(self)

**Description:** Return all items in the query params. If a key occurs more than once
only the first item for that key is returned.

Usage:

q = httpx.QueryParams("a=123&a=456&b=789")
assert list(q.items()) == [("a", "123"), ("b", "789")]

### Function: multi_items(self)

**Description:** Return all items in the query params. Allow duplicate keys to occur.

Usage:

q = httpx.QueryParams("a=123&a=456&b=789")
assert list(q.multi_items()) == [("a", "123"), ("a", "456"), ("b", "789")]

### Function: get(self, key, default)

**Description:** Get a value from the query param for a given key. If the key occurs
more than once, then only the first value is returned.

Usage:

q = httpx.QueryParams("a=123&a=456&b=789")
assert q.get("a") == "123"

### Function: get_list(self, key)

**Description:** Get all values from the query param for a given key.

Usage:

q = httpx.QueryParams("a=123&a=456&b=789")
assert q.get_list("a") == ["123", "456"]

### Function: set(self, key, value)

**Description:** Return a new QueryParams instance, setting the value of a key.

Usage:

q = httpx.QueryParams("a=123")
q = q.set("a", "456")
assert q == httpx.QueryParams("a=456")

### Function: add(self, key, value)

**Description:** Return a new QueryParams instance, setting or appending the value of a key.

Usage:

q = httpx.QueryParams("a=123")
q = q.add("a", "456")
assert q == httpx.QueryParams("a=123&a=456")

### Function: remove(self, key)

**Description:** Return a new QueryParams instance, removing the value of a key.

Usage:

q = httpx.QueryParams("a=123")
q = q.remove("a")
assert q == httpx.QueryParams("")

### Function: merge(self, params)

**Description:** Return a new QueryParams instance, updated with.

Usage:

q = httpx.QueryParams("a=123")
q = q.merge({"b": "456"})
assert q == httpx.QueryParams("a=123&b=456")

q = httpx.QueryParams("a=123")
q = q.merge({"a": "456", "b": "789"})
assert q == httpx.QueryParams("a=456&b=789")

### Function: __getitem__(self, key)

### Function: __contains__(self, key)

### Function: __iter__(self)

### Function: __len__(self)

### Function: __bool__(self)

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: __str__(self)

### Function: __repr__(self)

### Function: update(self, params)

### Function: __setitem__(self, key, value)
