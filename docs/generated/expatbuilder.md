## AI Summary

A file named expatbuilder.py.


## Class: DefusedExpatBuilder

**Description:** Defused document builder

## Class: DefusedExpatBuilderNS

**Description:** Defused document builder that supports namespaces.

### Function: parse(file, namespaces, forbid_dtd, forbid_entities, forbid_external)

**Description:** Parse a document, returning the resulting Document node.

'file' may be either a file name or an open file object.

### Function: parseString(string, namespaces, forbid_dtd, forbid_entities, forbid_external)

**Description:** Parse a document from a string, returning the resulting
Document node.

### Function: __init__(self, options, forbid_dtd, forbid_entities, forbid_external)

### Function: defused_start_doctype_decl(self, name, sysid, pubid, has_internal_subset)

### Function: defused_entity_decl(self, name, is_parameter_entity, value, base, sysid, pubid, notation_name)

### Function: defused_unparsed_entity_decl(self, name, base, sysid, pubid, notation_name)

### Function: defused_external_entity_ref_handler(self, context, base, sysid, pubid)

### Function: install(self, parser)

### Function: install(self, parser)

### Function: reset(self)
