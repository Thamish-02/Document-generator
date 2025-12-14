## AI Summary

A file named _status_codes.py.


## Class: codes

**Description:** HTTP status codes and reason phrases

Status codes from the following RFCs are all observed:

    * RFC 7231: Hypertext Transfer Protocol (HTTP/1.1), obsoletes 2616
    * RFC 6585: Additional HTTP Status Codes
    * RFC 3229: Delta encoding in HTTP
    * RFC 4918: HTTP Extensions for WebDAV, obsoletes 2518
    * RFC 5842: Binding Extensions to WebDAV
    * RFC 7238: Permanent Redirect
    * RFC 2295: Transparent Content Negotiation in HTTP
    * RFC 2774: An HTTP Extension Framework
    * RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2)
    * RFC 2324: Hyper Text Coffee Pot Control Protocol (HTCPCP/1.0)
    * RFC 7725: An HTTP Status Code to Report Legal Obstacles
    * RFC 8297: An HTTP Status Code for Indicating Hints
    * RFC 8470: Using Early Data in HTTP

### Function: __new__(cls, value, phrase)

### Function: __str__(self)

### Function: get_reason_phrase(cls, value)

### Function: is_informational(cls, value)

**Description:** Returns `True` for 1xx status codes, `False` otherwise.

### Function: is_success(cls, value)

**Description:** Returns `True` for 2xx status codes, `False` otherwise.

### Function: is_redirect(cls, value)

**Description:** Returns `True` for 3xx status codes, `False` otherwise.

### Function: is_client_error(cls, value)

**Description:** Returns `True` for 4xx status codes, `False` otherwise.

### Function: is_server_error(cls, value)

**Description:** Returns `True` for 5xx status codes, `False` otherwise.

### Function: is_error(cls, value)

**Description:** Returns `True` for 4xx or 5xx status codes, `False` otherwise.
