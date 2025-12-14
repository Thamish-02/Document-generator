## AI Summary

A file named ref_resolver.py.


### Function: get_id(schema)

**Description:** Originally ID was `id` and since v7 it's `$id`.

### Function: resolve_path(schema, fragment)

**Description:** Return definition from path.

Path is unescaped according https://tools.ietf.org/html/rfc6901

### Function: normalize(uri)

### Function: resolve_remote(uri, handlers)

**Description:** Resolve a remote ``uri``.

.. note::

    urllib library is used to fetch requests from the remote ``uri``
    if handlers does notdefine otherwise.

## Class: RefResolver

**Description:** Resolve JSON References.

### Function: __init__(self, base_uri, schema, store, cache, handlers)

**Description:** `base_uri` is URI of the referring document from the `schema`.
`store` is an dictionary that will be used to cache the fetched schemas
(if `cache=True`).

Please notice that you can have caching problems when compiling schemas
with colliding `$ref`. To force overwriting use `cache=False` or
explicitly pass the `store` argument (with a brand new dictionary)

### Function: from_schema(cls, schema, handlers)

**Description:** Construct a resolver from a JSON schema object.

### Function: in_scope(self, scope)

**Description:** Context manager to handle current scope.

### Function: resolving(self, ref)

**Description:** Context manager which resolves a JSON ``ref`` and enters the
resolution scope of this ref.

### Function: get_uri(self)

### Function: get_scope_name(self)

**Description:** Get current scope and return it as a valid function name.

### Function: walk(self, node)

**Description:** Walk thru schema and dereferencing ``id`` and ``$ref`` instances
