## AI Summary

A file named alphabeticalattributes.py.


### Function: _attr_key(attr)

**Description:** Return an appropriate key for an attribute for sorting

Attributes have a namespace that can be either ``None`` or a string. We
can't compare the two because they're different types, so we convert
``None`` to an empty string first.

## Class: Filter

**Description:** Alphabetizes attributes for elements

### Function: __iter__(self)
