## AI Summary

A file named schema.py.


## Class: EventSchemaUnrecognized

**Description:** An error for an unrecognized event schema.

## Class: EventSchemaLoadingError

**Description:** An error for an event schema loading error.

## Class: EventSchemaFileAbsent

**Description:** An error for an absent event schema file.

## Class: EventSchema

**Description:** A validated schema that can be used.

On instantiation, validate the schema against
Jupyter Event's metaschema.

Parameters
----------
schema: dict or str
    JSON schema to validate against Jupyter Events.

validator_class: jsonschema.validators
    The validator class from jsonschema used to validate instances
    of this event schema. The schema itself will be validated
    against Jupyter Event's metaschema to ensure that
    any schema registered here follows the expected form
    of Jupyter Events.

registry:
    Registry for nested JSON schema references.

### Function: __init__(self, schema, validator_class, format_checker, registry)

**Description:** Initialize an event schema.

### Function: __repr__(self)

**Description:** A string repr for an event schema.

### Function: _ensure_yaml_loaded(schema, was_str)

**Description:** Ensures schema was correctly loaded into a dictionary. Raises
EventSchemaLoadingError otherwise.

### Function: _load_schema(schema)

**Description:** Load a JSON schema from different sources/data types.

`schema` could be a dictionary or serialized string representing the
schema itself or a Pathlib object representing a schema file on disk.

Returns a dictionary with schema data.

### Function: id(self)

**Description:** Schema $id field.

### Function: version(self)

**Description:** Schema's version.

### Function: properties(self)

### Function: validate(self, data)

**Description:** Validate an incoming instance of this event schema.

### Function: intended_as_path(schema)
