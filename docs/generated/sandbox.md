## AI Summary

A file named sandbox.py.


### Function: safe_range()

**Description:** A range that can't generate ranges with a length of more than
MAX_RANGE items.

### Function: unsafe(f)

**Description:** Marks a function or method as unsafe.

.. code-block: python

    @unsafe
    def delete(self):
        pass

### Function: is_internal_attribute(obj, attr)

**Description:** Test if the attribute given is an internal python attribute.  For
example this function returns `True` for the `func_code` attribute of
python objects.  This is useful if the environment method
:meth:`~SandboxedEnvironment.is_safe_attribute` is overridden.

>>> from jinja2.sandbox import is_internal_attribute
>>> is_internal_attribute(str, "mro")
True
>>> is_internal_attribute(str, "upper")
False

### Function: modifies_known_mutable(obj, attr)

**Description:** This function checks if an attribute on a builtin mutable object
(list, dict, set or deque) or the corresponding ABCs would modify it
if called.

>>> modifies_known_mutable({}, "clear")
True
>>> modifies_known_mutable({}, "keys")
False
>>> modifies_known_mutable([], "append")
True
>>> modifies_known_mutable([], "index")
False

If called with an unsupported object, ``False`` is returned.

>>> modifies_known_mutable("foo", "upper")
False

## Class: SandboxedEnvironment

**Description:** The sandboxed environment.  It works like the regular environment but
tells the compiler to generate sandboxed code.  Additionally subclasses of
this environment may override the methods that tell the runtime what
attributes or functions are safe to access.

If the template tries to access insecure code a :exc:`SecurityError` is
raised.  However also other exceptions may occur during the rendering so
the caller has to ensure that all exceptions are caught.

## Class: ImmutableSandboxedEnvironment

**Description:** Works exactly like the regular `SandboxedEnvironment` but does not
permit modifications on the builtin mutable objects `list`, `set`, and
`dict` by using the :func:`modifies_known_mutable` function.

## Class: SandboxedFormatter

## Class: SandboxedEscapeFormatter

### Function: __init__(self)

### Function: is_safe_attribute(self, obj, attr, value)

**Description:** The sandboxed environment will call this method to check if the
attribute of an object is safe to access.  Per default all attributes
starting with an underscore are considered private as well as the
special attributes of internal python objects as returned by the
:func:`is_internal_attribute` function.

### Function: is_safe_callable(self, obj)

**Description:** Check if an object is safely callable. By default callables
are considered safe unless decorated with :func:`unsafe`.

This also recognizes the Django convention of setting
``func.alters_data = True``.

### Function: call_binop(self, context, operator, left, right)

**Description:** For intercepted binary operator calls (:meth:`intercepted_binops`)
this function is executed instead of the builtin operator.  This can
be used to fine tune the behavior of certain operators.

.. versionadded:: 2.6

### Function: call_unop(self, context, operator, arg)

**Description:** For intercepted unary operator calls (:meth:`intercepted_unops`)
this function is executed instead of the builtin operator.  This can
be used to fine tune the behavior of certain operators.

.. versionadded:: 2.6

### Function: getitem(self, obj, argument)

**Description:** Subscribe an object from sandboxed code.

### Function: getattr(self, obj, attribute)

**Description:** Subscribe an object from sandboxed code and prefer the
attribute.  The attribute passed *must* be a bytestring.

### Function: unsafe_undefined(self, obj, attribute)

**Description:** Return an undefined object for unsafe attributes.

### Function: wrap_str_format(self, value)

**Description:** If the given value is a ``str.format`` or ``str.format_map`` method,
return a new function than handles sandboxing. This is done at access
rather than in :meth:`call`, so that calls made without ``call`` are
also sandboxed.

### Function: call(__self, __context, __obj)

**Description:** Call an object from sandboxed code.

### Function: is_safe_attribute(self, obj, attr, value)

### Function: __init__(self, env)

### Function: get_field(self, field_name, args, kwargs)

### Function: wrapper()
