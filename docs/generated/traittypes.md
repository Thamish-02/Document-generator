## AI Summary

A file named traittypes.py.


## Class: TypeFromClasses

**Description:** A trait whose value must be a subclass of a class in a specified list of classes.

## Class: InstanceFromClasses

**Description:** A trait whose value must be an instance of a class in a specified list of classes.
The value can also be an instance of a subclass of the specified classes.
Subclasses can declare default classes by overriding the klass attribute

### Function: __init__(self, default_value, klasses)

**Description:** Construct a Type trait
A Type trait specifies that its values must be subclasses of
a class in a list of possible classes.
If only ``default_value`` is given, it is used for the ``klasses`` as
well. If neither are given, both default to ``object``.
Parameters
----------
default_value : class, str or None
    The default value must be a subclass of klass.  If an str,
    the str must be a fully specified class name, like 'foo.bar.Bah'.
    The string is resolved into real class, when the parent
    :class:`HasTraits` class is instantiated.
klasses : list of class, str [ default object ]
    Values of this trait must be a subclass of klass.  The klass
    may be specified in a string like: 'foo.bar.MyClass'.
    The string is resolved into real class, when the parent
    :class:`HasTraits` class is instantiated.
allow_none : bool [ default False ]
    Indicates whether None is allowed as an assignable value.

### Function: subclass_from_klasses(self, value)

**Description:** Check that a given class is a subclasses found in the klasses list.

### Function: validate(self, obj, value)

**Description:** Validates that the value is a valid object instance.

### Function: info(self)

**Description:** Returns a description of the trait.

### Function: instance_init(self, obj)

**Description:** Initialize an instance.

### Function: _resolve_classes(self)

**Description:** Resolve all string names to actual classes.

### Function: default_value_repr(self)

**Description:** The default value repr.

### Function: __init__(self, klasses, args, kw)

**Description:** Construct an Instance trait.
This trait allows values that are instances of a particular
class or its subclasses.  Our implementation is quite different
from that of enthough.traits as we don't allow instances to be used
for klass and we handle the ``args`` and ``kw`` arguments differently.
Parameters
----------
klasses : list of classes or class_names (str)
    The class that forms the basis for the trait.  Class names
    can also be specified as strings, like 'foo.bar.Bar'.
args : tuple
    Positional arguments for generating the default value.
kw : dict
    Keyword arguments for generating the default value.
allow_none : bool [ default False ]
    Indicates whether None is allowed as a value.
Notes
-----
If both ``args`` and ``kw`` are None, then the default value is None.
If ``args`` is a tuple and ``kw`` is a dict, then the default is
created as ``klass(*args, **kw)``.  If exactly one of ``args`` or ``kw`` is
None, the None is replaced by ``()`` or ``{}``, respectively.

### Function: instance_from_importable_klasses(self, value)

**Description:** Check that a given class is a subclasses found in the klasses list.

### Function: validate(self, obj, value)

**Description:** Validate an instance.

### Function: info(self)

**Description:** Get the trait info.

### Function: instance_init(self, obj)

**Description:** Initialize the trait.

### Function: _resolve_classes(self)

**Description:** Resolve all string names to actual classes.

### Function: make_dynamic_default(self)

**Description:** Make the dynamic default for the trait.

### Function: default_value_repr(self)

**Description:** Get the default value repr.

### Function: from_string(self, s)

**Description:** Convert from a string.
