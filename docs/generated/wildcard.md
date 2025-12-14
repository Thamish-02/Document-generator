## AI Summary

A file named wildcard.py.


### Function: create_typestr2type_dicts(dont_include_in_type2typestr)

**Description:** Return dictionaries mapping lower case typename (e.g. 'tuple') to type
objects from the types package, and vice versa.

### Function: is_type(obj, typestr_or_type)

**Description:** is_type(obj, typestr_or_type) verifies if obj is of a certain type. It
can take strings or actual python types for the second argument, i.e.
'tuple'<->TupleType. 'all' matches all types.

TODO: Should be extended for choosing more than one type.

### Function: show_hidden(str, show_all)

**Description:** Return true for strings starting with single _ if show_all is true.

### Function: dict_dir(obj)

**Description:** Produce a dictionary of an object's attributes. Builds on dir2 by
checking that a getattr() call actually succeeds.

### Function: filter_ns(ns, name_pattern, type_pattern, ignore_case, show_all)

**Description:** Filter a namespace dictionary by name pattern and item type.

### Function: list_namespace(namespace, type_pattern, filter, ignore_case, show_all)

**Description:** Return dictionary of all objects in a namespace dictionary that match
type_pattern and filter.
