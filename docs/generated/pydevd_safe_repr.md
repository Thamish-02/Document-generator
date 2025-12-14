## AI Summary

A file named pydevd_safe_repr.py.


## Class: SafeRepr

### Function: __call__(self, obj)

**Description:** :param object obj:
    The object for which we want a representation.

:return str:
    Returns bytes encoded as utf-8 on py2 and str on py3.

### Function: _repr(self, obj, level)

**Description:** Returns an iterable of the parts in the final repr string.

### Function: _is_long_iter(self, obj, level)

### Function: _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element)

### Function: _repr_long_iter(self, obj)

### Function: _repr_dict(self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix)

### Function: _repr_str(self, obj, level)

### Function: _repr_other(self, obj, level)

### Function: _repr_obj(self, obj, level, limit_inner, limit_outer)

### Function: _convert_to_unicode_or_bytes_repr(self, obj_repr)

### Function: _bytes_as_unicode_if_possible(self, obj_repr)

### Function: has_obj_repr(t)
