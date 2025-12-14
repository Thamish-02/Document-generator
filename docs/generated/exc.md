## AI Summary

A file named exc.py.


## Class: BadData

**Description:** Raised if bad data of any sort was encountered. This is the base
for all exceptions that ItsDangerous defines.

.. versionadded:: 0.15

## Class: BadSignature

**Description:** Raised if a signature does not match.

## Class: BadTimeSignature

**Description:** Raised if a time-based signature is invalid. This is a subclass
of :class:`BadSignature`.

## Class: SignatureExpired

**Description:** Raised if a signature timestamp is older than ``max_age``. This
is a subclass of :exc:`BadTimeSignature`.

## Class: BadHeader

**Description:** Raised if a signed header is invalid in some form. This only
happens for serializers that have a header that goes with the
signature.

.. versionadded:: 0.24

## Class: BadPayload

**Description:** Raised if a payload is invalid. This could happen if the payload
is loaded despite an invalid signature, or if there is a mismatch
between the serializer and deserializer. The original exception
that occurred during loading is stored on as :attr:`original_error`.

.. versionadded:: 0.15

### Function: __init__(self, message)

### Function: __str__(self)

### Function: __init__(self, message, payload)

### Function: __init__(self, message, payload, date_signed)

### Function: __init__(self, message, payload, header, original_error)

### Function: __init__(self, message, original_error)
