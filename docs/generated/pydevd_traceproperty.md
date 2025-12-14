## AI Summary

A file named pydevd_traceproperty.py.


### Function: replace_builtin_property(new_property)

## Class: DebugProperty

**Description:** A custom property which allows python property to get
controlled by the debugger and selectively disable/re-enable
the tracing.

### Function: __init__(self, fget, fset, fdel, doc)

### Function: __get__(self, obj, objtype)

### Function: __set__(self, obj, value)

### Function: __delete__(self, obj)

### Function: getter(self, fget)

**Description:** Overriding getter decorator for the property

### Function: setter(self, fset)

**Description:** Overriding setter decorator for the property

### Function: deleter(self, fdel)

**Description:** Overriding deleter decorator for the property
