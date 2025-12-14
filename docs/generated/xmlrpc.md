## AI Summary

A file named xmlrpc.py.


### Function: defused_gzip_decode(data, limit)

**Description:** gzip encoded data -> unencoded data

Decode data using the gzip content encoding as described in RFC 1952

## Class: DefusedGzipDecodedResponse

**Description:** a file-like object to decode a response encoded with the gzip
method, as described in RFC 1952.

## Class: DefusedExpatParser

### Function: monkey_patch()

### Function: unmonkey_patch()

### Function: __init__(self, response, limit)

### Function: read(self, n)

### Function: close(self)

### Function: __init__(self, target, forbid_dtd, forbid_entities, forbid_external)

### Function: defused_start_doctype_decl(self, name, sysid, pubid, has_internal_subset)

### Function: defused_entity_decl(self, name, is_parameter_entity, value, base, sysid, pubid, notation_name)

### Function: defused_unparsed_entity_decl(self, name, base, sysid, pubid, notation_name)

### Function: defused_external_entity_ref_handler(self, context, base, sysid, pubid)
