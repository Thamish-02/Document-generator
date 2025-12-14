## AI Summary

A file named _typedattr.py.


### Function: typed_attribute()

**Description:** Return a unique object, used to mark typed attributes.

## Class: TypedAttributeSet

**Description:** Superclass for typed attribute collections.

Checks that every public attribute of every subclass has a type annotation.

## Class: TypedAttributeProvider

**Description:** Base class for classes that wish to provide typed extra attributes.

### Function: __init_subclass__(cls)

### Function: extra_attributes(self)

**Description:** A mapping of the extra attributes to callables that return the corresponding
values.

If the provider wraps another provider, the attributes from that wrapper should
also be included in the returned mapping (but the wrapper may override the
callables from the wrapped instance).

### Function: extra(self, attribute)

### Function: extra(self, attribute, default)

### Function: extra(self, attribute, default)

**Description:** extra(attribute, default=undefined)

Return the value of the given typed extra attribute.

:param attribute: the attribute (member of a :class:`~TypedAttributeSet`) to
    look for
:param default: the value that should be returned if no value is found for the
    attribute
:raises ~anyio.TypedAttributeLookupError: if the search failed and no default
    value was given
