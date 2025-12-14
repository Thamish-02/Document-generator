## AI Summary

A file named namespace.py.


## Class: ImplicitNSName

**Description:** Accessing names for implicit namespace packages should infer to nothing.
This object will prevent Jedi from raising exceptions

## Class: ImplicitNamespaceValue

**Description:** Provides support for implicit namespace packages

### Function: __init__(self, implicit_ns_value, string_name)

### Function: __init__(self, inference_state, string_names, paths)

### Function: get_filters(self, origin_scope)

### Function: get_qualified_names(self)

### Function: name(self)

### Function: py__file__(self)

### Function: py__package__(self)

**Description:** Return the fullname
        

### Function: py__path__(self)

### Function: py__name__(self)

### Function: is_namespace(self)

### Function: is_stub(self)

### Function: is_package(self)

### Function: as_context(self)

### Function: __repr__(self)
