## AI Summary

A file named url_safe.py.


## Class: URLSafeSerializerMixin

**Description:** Mixed in with a regular serializer it will attempt to zlib
compress the string to make it shorter if necessary. It will also
base64 encode the string so that it can safely be placed in a URL.

## Class: URLSafeSerializer

**Description:** Works like :class:`.Serializer` but dumps and loads into a URL
safe string consisting of the upper and lowercase character of the
alphabet as well as ``'_'``, ``'-'`` and ``'.'``.

## Class: URLSafeTimedSerializer

**Description:** Works like :class:`.TimedSerializer` but dumps and loads into a
URL safe string consisting of the upper and lowercase character of
the alphabet as well as ``'_'``, ``'-'`` and ``'.'``.

### Function: load_payload(self, payload)

### Function: dump_payload(self, obj)
