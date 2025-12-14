## AI Summary

A file named json.py.


## Class: JsonEncoder

**Description:** Customizable JSON encoder.

If the object implements __getstate__, then that method is invoked, and its
result is serialized instead of the object itself.

## Class: JsonObject

**Description:** A wrapped Python object that formats itself as JSON when asked for a string
representation via str() or format().

### Function: _converter(value, classinfo)

**Description:** Convert value (str) to number, otherwise return None if is not possible

### Function: of_type()

**Description:** Returns a validator for a JSON property that requires it to have a value of
the specified type. If optional=True, () is also allowed.

The meaning of classinfo is the same as for isinstance().

### Function: default(default)

**Description:** Returns a validator for a JSON property with a default value.

The validator will only allow property values that have the same type as the
specified default value.

### Function: enum()

**Description:** Returns a validator for a JSON enum.

The validator will only allow the property to have one of the specified values.

If optional=True, and the property is missing, the first value specified is used
as the default.

### Function: array(validate_item, vectorize, size)

**Description:** Returns a validator for a JSON array.

If the property is missing, it is treated as if it were []. Otherwise, it must
be a list.

If validate_item=False, it's treated as if it were (lambda x: x) - i.e. any item
is considered valid, and is unchanged. If validate_item is a type or a tuple,
it's treated as if it were json.of_type(validate).

Every item in the list is replaced with validate_item(item) in-place, propagating
any exceptions raised by the latter. If validate_item is a type or a tuple, it is
treated as if it were json.of_type(validate_item).

If vectorize=True, and the value is neither a list nor a dict, it is treated as
if it were a single-element list containing that single value - e.g. "foo" is
then the same as ["foo"]; but {} is an error, and not [{}].

If size is not None, it can be an int, a tuple of one int, a tuple of two ints,
or a set. If it's an int, the array must have exactly that many elements. If it's
a tuple of one int, it's the minimum length. If it's a tuple of two ints, they
are the minimum and the maximum lengths. If it's a set, it's the set of sizes that
are valid - e.g. for {2, 4}, the array can be either 2 or 4 elements long.

### Function: object(validate_value)

**Description:** Returns a validator for a JSON object.

If the property is missing, it is treated as if it were {}. Otherwise, it must
be a dict.

If validate_value=False, it's treated as if it were (lambda x: x) - i.e. any
value is considered valid, and is unchanged. If validate_value is a type or a
tuple, it's treated as if it were json.of_type(validate_value).

Every value in the dict is replaced with validate_value(value) in-place, propagating
any exceptions raised by the latter. If validate_value is a type or a tuple, it is
treated as if it were json.of_type(validate_value). Keys are not affected.

### Function: repr(value)

### Function: default(self, value)

### Function: __init__(self, value)

### Function: __getstate__(self)

### Function: __repr__(self)

### Function: __str__(self)

### Function: __format__(self, format_spec)

**Description:** If format_spec is empty, uses self.json_encoder to serialize self.value
as a string. Otherwise, format_spec is treated as an argument list to be
passed to self.json_encoder_factory - which defaults to JSONEncoder - and
then the resulting formatter is used to serialize self.value as a string.

Example::

    format("{0} {0:indent=4,sort_keys=True}", json.repr(x))

### Function: validate(value)

### Function: validate(value)

### Function: validate(value)

### Function: validate(value)

### Function: validate(value)
