## AI Summary

A file named _legacy_keywords.py.


### Function: ignore_ref_siblings(schema)

**Description:** Ignore siblings of ``$ref`` if it is present.

Otherwise, return all keywords.

Suitable for use with `create`'s ``applicable_validators`` argument.

### Function: dependencies_draft3(validator, dependencies, instance, schema)

### Function: dependencies_draft4_draft6_draft7(validator, dependencies, instance, schema)

**Description:** Support for the ``dependencies`` keyword from pre-draft 2019-09.

In later drafts, the keyword was split into separate
``dependentRequired`` and ``dependentSchemas`` validators.

### Function: disallow_draft3(validator, disallow, instance, schema)

### Function: extends_draft3(validator, extends, instance, schema)

### Function: items_draft3_draft4(validator, items, instance, schema)

### Function: additionalItems(validator, aI, instance, schema)

### Function: items_draft6_draft7_draft201909(validator, items, instance, schema)

### Function: minimum_draft3_draft4(validator, minimum, instance, schema)

### Function: maximum_draft3_draft4(validator, maximum, instance, schema)

### Function: properties_draft3(validator, properties, instance, schema)

### Function: type_draft3(validator, types, instance, schema)

### Function: contains_draft6_draft7(validator, contains, instance, schema)

### Function: recursiveRef(validator, recursiveRef, instance, schema)

### Function: find_evaluated_item_indexes_by_schema(validator, instance, schema)

**Description:** Get all indexes of items that get evaluated under the current schema.

Covers all keywords related to unevaluatedItems: items, prefixItems, if,
then, else, contains, unevaluatedItems, allOf, oneOf, anyOf

### Function: unevaluatedItems_draft2019(validator, unevaluatedItems, instance, schema)

### Function: find_evaluated_property_keys_by_schema(validator, instance, schema)

### Function: unevaluatedProperties_draft2019(validator, uP, instance, schema)
