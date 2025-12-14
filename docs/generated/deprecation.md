## AI Summary

A file named deprecation.py.


## Class: MatplotlibDeprecationWarning

**Description:** A class for issuing deprecation warnings for Matplotlib users.

### Function: _generate_deprecation_warning(since, message, name, alternative, pending, obj_type, addendum)

### Function: warn_deprecated(since)

**Description:** Display a standardized deprecation.

Parameters
----------
since : str
    The release at which this API became deprecated.
message : str, optional
    Override the default deprecation message.  The ``%(since)s``,
    ``%(name)s``, ``%(alternative)s``, ``%(obj_type)s``, ``%(addendum)s``,
    and ``%(removal)s`` format specifiers will be replaced by the values
    of the respective arguments passed to this function.
name : str, optional
    The name of the deprecated object.
alternative : str, optional
    An alternative API that the user may use in place of the deprecated
    API.  The deprecation warning will tell the user about this alternative
    if provided.
pending : bool, optional
    If True, uses a PendingDeprecationWarning instead of a
    DeprecationWarning.  Cannot be used together with *removal*.
obj_type : str, optional
    The object type being deprecated.
addendum : str, optional
    Additional text appended directly to the final message.
removal : str, optional
    The expected removal version.  With the default (an empty string), a
    removal version is automatically computed from *since*.  Set to other
    Falsy values to not schedule a removal date.  Cannot be used together
    with *pending*.

Examples
--------
::

    # To warn of the deprecation of "matplotlib.name_of_module"
    warn_deprecated('1.4.0', name='matplotlib.name_of_module',
                    obj_type='module')

### Function: deprecated(since)

**Description:** Decorator to mark a function, a class, or a property as deprecated.

When deprecating a classmethod, a staticmethod, or a property, the
``@deprecated`` decorator should go *under* ``@classmethod`` and
``@staticmethod`` (i.e., `deprecated` should directly decorate the
underlying callable), but *over* ``@property``.

When deprecating a class ``C`` intended to be used as a base class in a
multiple inheritance hierarchy, ``C`` *must* define an ``__init__`` method
(if ``C`` instead inherited its ``__init__`` from its own base class, then
``@deprecated`` would mess up ``__init__`` inheritance when installing its
own (deprecation-emitting) ``C.__init__``).

Parameters are the same as for `warn_deprecated`, except that *obj_type*
defaults to 'class' if decorating a class, 'attribute' if decorating a
property, and 'function' otherwise.

Examples
--------
::

    @deprecated('1.4.0')
    def the_function_to_deprecate():
        pass

## Class: deprecate_privatize_attribute

**Description:** Helper to deprecate public access to an attribute (or method).

This helper should only be used at class scope, as follows::

    class Foo:
        attr = _deprecate_privatize_attribute(*args, **kwargs)

where *all* parameters are forwarded to `deprecated`.  This form makes
``attr`` a property which forwards read and write access to ``self._attr``
(same name but with a leading underscore), with a deprecation warning.
Note that the attribute name is derived from *the name this helper is
assigned to*.  This helper also works for deprecating methods.

### Function: rename_parameter(since, old, new, func)

**Description:** Decorator indicating that parameter *old* of *func* is renamed to *new*.

The actual implementation of *func* should use *new*, not *old*.  If *old*
is passed to *func*, a DeprecationWarning is emitted, and its value is
used, even if *new* is also passed by keyword (this is to simplify pyplot
wrapper functions, which always pass *new* explicitly to the Axes method).
If *new* is also passed but positionally, a TypeError will be raised by the
underlying function during argument binding.

Examples
--------
::

    @_api.rename_parameter("3.1", "bad_name", "good_name")
    def func(good_name): ...

## Class: _deprecated_parameter_class

### Function: delete_parameter(since, name, func)

**Description:** Decorator indicating that parameter *name* of *func* is being deprecated.

The actual implementation of *func* should keep the *name* parameter in its
signature, or accept a ``**kwargs`` argument (through which *name* would be
passed).

Parameters that come after the deprecated parameter effectively become
keyword-only (as they cannot be passed positionally without triggering the
DeprecationWarning on the deprecated parameter), and should be marked as
such after the deprecation period has passed and the deprecated parameter
is removed.

Parameters other than *since*, *name*, and *func* are keyword-only and
forwarded to `.warn_deprecated`.

Examples
--------
::

    @_api.delete_parameter("3.1", "unused")
    def func(used_arg, other_arg, unused, more_args): ...

### Function: make_keyword_only(since, name, func)

**Description:** Decorator indicating that passing parameter *name* (or any of the following
ones) positionally to *func* is being deprecated.

When used on a method that has a pyplot wrapper, this should be the
outermost decorator, so that :file:`boilerplate.py` can access the original
signature.

### Function: deprecate_method_override(method, obj)

**Description:** Return ``obj.method`` with a deprecation if it was overridden, else None.

Parameters
----------
method
    An unbound method, i.e. an expression of the form
    ``Class.method_name``.  Remember that within the body of a method, one
    can always use ``__class__`` to refer to the class that is currently
    being defined.
obj
    Either an object of the class where *method* is defined, or a subclass
    of that class.
allow_empty : bool, default: False
    Whether to allow overrides by "empty" methods without emitting a
    warning.
**kwargs
    Additional parameters passed to `warn_deprecated` to generate the
    deprecation warning; must at least include the "since" key.

### Function: suppress_matplotlib_deprecation_warning()

### Function: deprecate(obj, message, name, alternative, pending, obj_type, addendum)

### Function: __init__(self)

### Function: __set_name__(self, owner, name)

### Function: wrapper()

### Function: __repr__(self)

### Function: wrapper()

### Function: wrapper()

### Function: empty()

### Function: empty_with_docstring()

**Description:** doc

### Function: emit_warning()

### Function: wrapper()

### Function: finalize(wrapper, new_doc)

## Class: _deprecated_property

### Function: finalize(_, new_doc)

### Function: finalize(wrapper, new_doc)

### Function: __get__(self, instance, owner)

### Function: __set__(self, instance, value)

### Function: __delete__(self, instance)

### Function: __set_name__(self, owner, set_name)
