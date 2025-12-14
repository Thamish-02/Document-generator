## AI Summary

A file named schema_registry.py.


## Class: SchemaRegistryException

**Description:** Exception class for Jupyter Events Schema Registry Errors.

## Class: SchemaRegistry

**Description:** A convenient API for storing and searching a group of schemas.

### Function: __init__(self, schemas)

**Description:** Initialize the registry.

### Function: __contains__(self, key)

**Description:** Syntax sugar to check if a schema is found in the registry

### Function: __repr__(self)

**Description:** The str repr of the registry.

### Function: _add(self, schema_obj)

### Function: schema_ids(self)

### Function: register(self, schema)

**Description:** Add a valid schema to the registry.

All schemas are validated against the Jupyter Events meta-schema
found here:

### Function: get(self, id_)

**Description:** Fetch a given schema. If the schema is not found,
this will raise a KeyError.

### Function: remove(self, id_)

**Description:** Remove a given schema. If the schema is not found,
this will raise a KeyError.

### Function: validate_event(self, id_, data)

**Description:** Validate an event against a schema within this
registry.
