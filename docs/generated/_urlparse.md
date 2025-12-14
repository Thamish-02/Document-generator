## AI Summary

A file named _urlparse.py.


## Class: ParseResult

### Function: urlparse(url)

### Function: encode_host(host)

### Function: normalize_port(port, scheme)

### Function: validate_path(path, has_scheme, has_authority)

**Description:** Path validation rules that depend on if the URL contains
a scheme or authority component.

See https://datatracker.ietf.org/doc/html/rfc3986.html#section-3.3

### Function: normalize_path(path)

**Description:** Drop "." and ".." segments from a URL path.

For example:

    normalize_path("/path/./to/somewhere/..") == "/path/to"

### Function: PERCENT(string)

### Function: percent_encoded(string, safe)

**Description:** Use percent-encoding to quote a string.

### Function: quote(string, safe)

**Description:** Use percent-encoding to quote a string, omitting existing '%xx' escape sequences.

See: https://www.rfc-editor.org/rfc/rfc3986#section-2.1

* `string`: The string to be percent-escaped.
* `safe`: A string containing characters that may be treated as safe, and do not
    need to be escaped. Unreserved characters are always treated as safe.
    See: https://www.rfc-editor.org/rfc/rfc3986#section-2.3

### Function: authority(self)

### Function: netloc(self)

### Function: copy_with(self)

### Function: __str__(self)
