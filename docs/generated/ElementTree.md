## AI Summary

A file named ElementTree.py.


### Function: _get_py3_cls()

**Description:** Python 3.3 hides the pure Python code but defusedxml requires it.

The code is based on test.support.import_fresh_module().

## Class: DefusedXMLParser

### Function: __init__(self, html, target, encoding, forbid_dtd, forbid_entities, forbid_external)

### Function: defused_start_doctype_decl(self, name, sysid, pubid, has_internal_subset)

### Function: defused_entity_decl(self, name, is_parameter_entity, value, base, sysid, pubid, notation_name)

### Function: defused_unparsed_entity_decl(self, name, base, sysid, pubid, notation_name)

### Function: defused_external_entity_ref_handler(self, context, base, sysid, pubid)
