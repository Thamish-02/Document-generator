## AI Summary

A file named _make.py.


## Class: _Nothing

**Description:** Sentinel to indicate the lack of a value when `None` is ambiguous.

If extending attrs, you can use ``typing.Literal[NOTHING]`` to show
that a value may be ``NOTHING``.

.. versionchanged:: 21.1.0 ``bool(NOTHING)`` is now False.
.. versionchanged:: 22.2.0 ``NOTHING`` is now an ``enum.Enum`` variant.

## Class: _CacheHashWrapper

**Description:** An integer subclass that pickles / copies as None

This is used for non-slots classes with ``cache_hash=True``, to avoid
serializing a potentially (even likely) invalid hash value. Since `None`
is the default value for uncalculated hashes, whenever this is copied,
the copy's value for the hash should automatically reset.

See GH #613 for more details.

### Function: attrib(default, validator, repr, cmp, hash, init, metadata, type, converter, factory, kw_only, eq, order, on_setattr, alias)

**Description:** Create a new field / attribute on a class.

Identical to `attrs.field`, except it's not keyword-only.

Consider using `attrs.field` in new code (``attr.ib`` will *never* go away,
though).

..  warning::

    Does **nothing** unless the class is also decorated with
    `attr.s` (or similar)!


.. versionadded:: 15.2.0 *convert*
.. versionadded:: 16.3.0 *metadata*
.. versionchanged:: 17.1.0 *validator* can be a ``list`` now.
.. versionchanged:: 17.1.0
   *hash* is `None` and therefore mirrors *eq* by default.
.. versionadded:: 17.3.0 *type*
.. deprecated:: 17.4.0 *convert*
.. versionadded:: 17.4.0
   *converter* as a replacement for the deprecated *convert* to achieve
   consistency with other noun-based arguments.
.. versionadded:: 18.1.0
   ``factory=f`` is syntactic sugar for ``default=attr.Factory(f)``.
.. versionadded:: 18.2.0 *kw_only*
.. versionchanged:: 19.2.0 *convert* keyword argument removed.
.. versionchanged:: 19.2.0 *repr* also accepts a custom callable.
.. deprecated:: 19.2.0 *cmp* Removal on or after 2021-06-01.
.. versionadded:: 19.2.0 *eq* and *order*
.. versionadded:: 20.1.0 *on_setattr*
.. versionchanged:: 20.3.0 *kw_only* backported to Python 2
.. versionchanged:: 21.1.0
   *eq*, *order*, and *cmp* also accept a custom callable
.. versionchanged:: 21.1.0 *cmp* undeprecated
.. versionadded:: 22.2.0 *alias*
.. versionchanged:: 25.4.0
   *kw_only* can now be None, and its default is also changed from False to
   None.

### Function: _compile_and_eval(script, globs, locs, filename)

**Description:** Evaluate the script with the given global (globs) and local (locs)
variables.

### Function: _linecache_and_compile(script, filename, globs, locals)

**Description:** Cache the script with _linecache_, compile it and return the _locals_.

### Function: _make_attr_tuple_class(cls_name, attr_names)

**Description:** Create a tuple subclass to hold `Attribute`s for an `attrs` class.

The subclass is a bare tuple with properties for names.

class MyClassAttributes(tuple):
    __slots__ = ()
    x = property(itemgetter(0))

## Class: _Attributes

### Function: _is_class_var(annot)

**Description:** Check whether *annot* is a typing.ClassVar.

The string comparison hack is used to avoid evaluating all string
annotations which would put attrs-based classes at a performance
disadvantage compared to plain old classes.

### Function: _has_own_attribute(cls, attrib_name)

**Description:** Check whether *cls* defines *attrib_name* (and doesn't just inherit it).

### Function: _collect_base_attrs(cls, taken_attr_names)

**Description:** Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.

### Function: _collect_base_attrs_broken(cls, taken_attr_names)

**Description:** Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.

N.B. *taken_attr_names* will be mutated.

Adhere to the old incorrect behavior.

Notably it collects from the front and considers inherited attributes which
leads to the buggy behavior reported in #428.

### Function: _transform_attrs(cls, these, auto_attribs, kw_only, collect_by_mro, field_transformer)

**Description:** Transform all `_CountingAttr`s on a class into `Attribute`s.

If *these* is passed, use that and don't look for them on the class.

If *collect_by_mro* is True, collect them in the correct MRO order,
otherwise use the old -- incorrect -- order.  See #428.

Return an `_Attributes`.

### Function: _make_cached_property_getattr(cached_properties, original_getattr, cls)

### Function: _frozen_setattrs(self, name, value)

**Description:** Attached to frozen classes as __setattr__.

### Function: _frozen_delattrs(self, name)

**Description:** Attached to frozen classes as __delattr__.

### Function: evolve()

**Description:** Create a new instance, based on the first positional argument with
*changes* applied.

.. tip::

   On Python 3.13 and later, you can also use `copy.replace` instead.

Args:

    inst:
        Instance of a class with *attrs* attributes. *inst* must be passed
        as a positional argument.

    changes:
        Keyword changes in the new copy.

Returns:
    A copy of inst with *changes* incorporated.

Raises:
    TypeError:
        If *attr_name* couldn't be found in the class ``__init__``.

    attrs.exceptions.NotAnAttrsClassError:
        If *cls* is not an *attrs* class.

.. versionadded:: 17.1.0
.. deprecated:: 23.1.0
   It is now deprecated to pass the instance using the keyword argument
   *inst*. It will raise a warning until at least April 2024, after which
   it will become an error. Always pass the instance as a positional
   argument.
.. versionchanged:: 24.1.0
   *inst* can't be passed as a keyword argument anymore.

## Class: _ClassBuilder

**Description:** Iteratively build *one* class.

### Function: _determine_attrs_eq_order(cmp, eq, order, default_eq)

**Description:** Validate the combination of *cmp*, *eq*, and *order*. Derive the effective
values of eq and order.  If *eq* is None, set it to *default_eq*.

### Function: _determine_attrib_eq_order(cmp, eq, order, default_eq)

**Description:** Validate the combination of *cmp*, *eq*, and *order*. Derive the effective
values of eq and order.  If *eq* is None, set it to *default_eq*.

### Function: _determine_whether_to_implement(cls, flag, auto_detect, dunders, default)

**Description:** Check whether we should implement a set of methods for *cls*.

*flag* is the argument passed into @attr.s like 'init', *auto_detect* the
same as passed into @attr.s and *dunders* is a tuple of attribute names
whose presence signal that the user has implemented it themselves.

Return *default* if no reason for either for or against is found.

### Function: attrs(maybe_cls, these, repr_ns, repr, cmp, hash, init, slots, frozen, weakref_slot, str, auto_attribs, kw_only, cache_hash, auto_exc, eq, order, auto_detect, collect_by_mro, getstate_setstate, on_setattr, field_transformer, match_args, unsafe_hash, force_kw_only)

**Description:** A class decorator that adds :term:`dunder methods` according to the
specified attributes using `attr.ib` or the *these* argument.

Consider using `attrs.define` / `attrs.frozen` in new code (``attr.s`` will
*never* go away, though).

Args:
    repr_ns (str):
        When using nested classes, there was no way in Python 2 to
        automatically detect that.  This argument allows to set a custom
        name for a more meaningful ``repr`` output.  This argument is
        pointless in Python 3 and is therefore deprecated.

.. caution::
    Refer to `attrs.define` for the rest of the parameters, but note that they
    can have different defaults.

    Notably, leaving *on_setattr* as `None` will **not** add any hooks.

.. versionadded:: 16.0.0 *slots*
.. versionadded:: 16.1.0 *frozen*
.. versionadded:: 16.3.0 *str*
.. versionadded:: 16.3.0 Support for ``__attrs_post_init__``.
.. versionchanged:: 17.1.0
   *hash* supports `None` as value which is also the default now.
.. versionadded:: 17.3.0 *auto_attribs*
.. versionchanged:: 18.1.0
   If *these* is passed, no attributes are deleted from the class body.
.. versionchanged:: 18.1.0 If *these* is ordered, the order is retained.
.. versionadded:: 18.2.0 *weakref_slot*
.. deprecated:: 18.2.0
   ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__`` now raise a
   `DeprecationWarning` if the classes compared are subclasses of
   each other. ``__eq`` and ``__ne__`` never tried to compared subclasses
   to each other.
.. versionchanged:: 19.2.0
   ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__`` now do not consider
   subclasses comparable anymore.
.. versionadded:: 18.2.0 *kw_only*
.. versionadded:: 18.2.0 *cache_hash*
.. versionadded:: 19.1.0 *auto_exc*
.. deprecated:: 19.2.0 *cmp* Removal on or after 2021-06-01.
.. versionadded:: 19.2.0 *eq* and *order*
.. versionadded:: 20.1.0 *auto_detect*
.. versionadded:: 20.1.0 *collect_by_mro*
.. versionadded:: 20.1.0 *getstate_setstate*
.. versionadded:: 20.1.0 *on_setattr*
.. versionadded:: 20.3.0 *field_transformer*
.. versionchanged:: 21.1.0
   ``init=False`` injects ``__attrs_init__``
.. versionchanged:: 21.1.0 Support for ``__attrs_pre_init__``
.. versionchanged:: 21.1.0 *cmp* undeprecated
.. versionadded:: 21.3.0 *match_args*
.. versionadded:: 22.2.0
   *unsafe_hash* as an alias for *hash* (for :pep:`681` compliance).
.. deprecated:: 24.1.0 *repr_ns*
.. versionchanged:: 24.1.0
   Instances are not compared as tuples of attributes anymore, but using a
   big ``and`` condition. This is faster and has more correct behavior for
   uncomparable values like `math.nan`.
.. versionadded:: 24.1.0
   If a class has an *inherited* classmethod called
   ``__attrs_init_subclass__``, it is executed after the class is created.
.. deprecated:: 24.1.0 *hash* is deprecated in favor of *unsafe_hash*.
.. versionchanged:: 25.4.0
   *kw_only* now only applies to attributes defined in the current class,
   and respects attribute-level ``kw_only=False`` settings.
.. versionadded:: 25.4.0 *force_kw_only*

### Function: _has_frozen_base_class(cls)

**Description:** Check whether *cls* has a frozen ancestor by looking at its
__setattr__.

### Function: _generate_unique_filename(cls, func_name)

**Description:** Create a "filename" suitable for a function being generated.

### Function: _make_hash_script(cls, attrs, frozen, cache_hash)

### Function: _add_hash(cls, attrs)

**Description:** Add a hash method to *cls*.

### Function: __ne__(self, other)

**Description:** Check equality and either forward a NotImplemented or
return the result negated.

### Function: _make_eq_script(attrs)

**Description:** Create __eq__ method for *cls* with *attrs*.

### Function: _make_order(cls, attrs)

**Description:** Create ordering methods for *cls* with *attrs*.

### Function: _add_eq(cls, attrs)

**Description:** Add equality methods to *cls* with *attrs*.

### Function: _make_repr_script(attrs, ns)

**Description:** Create the source and globs for a __repr__ and return it.

### Function: _add_repr(cls, ns, attrs)

**Description:** Add a repr method to *cls*.

### Function: fields(cls)

**Description:** Return the tuple of *attrs* attributes for a class.

The tuple also allows accessing the fields by their names (see below for
examples).

Args:
    cls (type): Class to introspect.

Raises:
    TypeError: If *cls* is not a class.

    attrs.exceptions.NotAnAttrsClassError:
        If *cls* is not an *attrs* class.

Returns:
    tuple (with name accessors) of `attrs.Attribute`

.. versionchanged:: 16.2.0 Returned tuple allows accessing the fields
   by name.
.. versionchanged:: 23.1.0 Add support for generic classes.

### Function: fields_dict(cls)

**Description:** Return an ordered dictionary of *attrs* attributes for a class, whose keys
are the attribute names.

Args:
    cls (type): Class to introspect.

Raises:
    TypeError: If *cls* is not a class.

    attrs.exceptions.NotAnAttrsClassError:
        If *cls* is not an *attrs* class.

Returns:
    dict[str, attrs.Attribute]: Dict of attribute name to definition

.. versionadded:: 18.1.0

### Function: validate(inst)

**Description:** Validate all attributes on *inst* that have a validator.

Leaves all exceptions through.

Args:
    inst: Instance of a class with *attrs* attributes.

### Function: _is_slot_attr(a_name, base_attr_map)

**Description:** Check if the attribute name comes from a slot class.

### Function: _make_init_script(cls, attrs, pre_init, pre_init_has_args, post_init, frozen, slots, cache_hash, base_attr_map, is_exc, cls_on_setattr, attrs_init)

### Function: _setattr(attr_name, value_var, has_on_setattr)

**Description:** Use the cached object.setattr to set *attr_name* to *value_var*.

### Function: _setattr_with_converter(attr_name, value_var, has_on_setattr, converter)

**Description:** Use the cached object.setattr to set *attr_name* to *value_var*, but run
its converter first.

### Function: _assign(attr_name, value, has_on_setattr)

**Description:** Unless *attr_name* has an on_setattr hook, use normal assignment. Otherwise
relegate to _setattr.

### Function: _assign_with_converter(attr_name, value_var, has_on_setattr, converter)

**Description:** Unless *attr_name* has an on_setattr hook, use normal assignment after
conversion. Otherwise relegate to _setattr_with_converter.

### Function: _determine_setters(frozen, slots, base_attr_map)

**Description:** Determine the correct setter functions based on whether a class is frozen
and/or slotted.

### Function: _attrs_to_init_script(attrs, is_frozen, is_slotted, call_pre_init, pre_init_has_args, call_post_init, does_cache_hash, base_attr_map, is_exc, needs_cached_setattr, has_cls_on_setattr, method_name)

**Description:** Return a script of an initializer for *attrs*, a dict of globals, and
annotations for the initializer.

The globals are required by the generated script.

### Function: _default_init_alias_for(name)

**Description:** The default __init__ parameter name for a field.

This performs private-name adjustment via leading-unscore stripping,
and is the default value of Attribute.alias if not provided.

## Class: Attribute

**Description:** *Read-only* representation of an attribute.

.. warning::

   You should never instantiate this class yourself.

The class has *all* arguments of `attr.ib` (except for ``factory`` which is
only syntactic sugar for ``default=Factory(...)`` plus the following:

- ``name`` (`str`): The name of the attribute.
- ``alias`` (`str`): The __init__ parameter name of the attribute, after
  any explicit overrides and default private-attribute-name handling.
- ``inherited`` (`bool`): Whether or not that attribute has been inherited
  from a base class.
- ``eq_key`` and ``order_key`` (`typing.Callable` or `None`): The
  callables that are used for comparing and ordering objects by this
  attribute, respectively. These are set by passing a callable to
  `attr.ib`'s ``eq``, ``order``, or ``cmp`` arguments. See also
  :ref:`comparison customization <custom-comparison>`.

Instances of this class are frequently used for introspection purposes
like:

- `fields` returns a tuple of them.
- Validators get them passed as the first argument.
- The :ref:`field transformer <transform-fields>` hook receives a list of
  them.
- The ``alias`` property exposes the __init__ parameter name of the field,
  with any overrides and default private-attribute handling applied.


.. versionadded:: 20.1.0 *inherited*
.. versionadded:: 20.1.0 *on_setattr*
.. versionchanged:: 20.2.0 *inherited* is not taken into account for
    equality checks and hashing anymore.
.. versionadded:: 21.1.0 *eq_key* and *order_key*
.. versionadded:: 22.2.0 *alias*

For the full version history of the fields, see `attr.ib`.

## Class: _CountingAttr

**Description:** Intermediate representation of attributes that uses a counter to preserve
the order in which the attributes have been defined.

*Internal* data structure of the attrs library.  Running into is most
likely the result of a bug like a forgotten `@attr.s` decorator.

## Class: ClassProps

**Description:** Effective class properties as derived from parameters to `attr.s()` or
`define()` decorators.

This is the same data structure that *attrs* uses internally to decide how
to construct the final class.

Warning:

    This feature is currently **experimental** and is not covered by our
    strict backwards-compatibility guarantees.


Attributes:
    is_exception (bool):
        Whether the class is treated as an exception class.

    is_slotted (bool):
        Whether the class is `slotted <slotted classes>`.

    has_weakref_slot (bool):
        Whether the class has a slot for weak references.

    is_frozen (bool):
        Whether the class is frozen.

    kw_only (KeywordOnly):
        Whether / how the class enforces keyword-only arguments on the
        ``__init__`` method.

    collected_fields_by_mro (bool):
        Whether the class fields were collected by method resolution order.
        That is, correctly but unlike `dataclasses`.

    added_init (bool):
        Whether the class has an *attrs*-generated ``__init__`` method.

    added_repr (bool):
        Whether the class has an *attrs*-generated ``__repr__`` method.

    added_eq (bool):
        Whether the class has *attrs*-generated equality methods.

    added_ordering (bool):
        Whether the class has *attrs*-generated ordering methods.

    hashability (Hashability): How `hashable <hashing>` the class is.

    added_match_args (bool):
        Whether the class supports positional `match <match>` over its
        fields.

    added_str (bool):
        Whether the class has an *attrs*-generated ``__str__`` method.

    added_pickling (bool):
        Whether the class has *attrs*-generated ``__getstate__`` and
        ``__setstate__`` methods for `pickle`.

    on_setattr_hook (Callable[[Any, Attribute[Any], Any], Any] | None):
        The class's ``__setattr__`` hook.

    field_transformer (Callable[[Attribute[Any]], Attribute[Any]] | None):
        The class's `field transformers <transform-fields>`.

.. versionadded:: 25.4.0

## Class: Factory

**Description:** Stores a factory callable.

If passed as the default value to `attrs.field`, the factory is used to
generate a new value.

Args:
    factory (typing.Callable):
        A callable that takes either none or exactly one mandatory
        positional argument depending on *takes_self*.

    takes_self (bool):
        Pass the partially initialized instance that is being initialized
        as a positional argument.

.. versionadded:: 17.1.0  *takes_self*

## Class: Converter

**Description:** Stores a converter callable.

Allows for the wrapped converter to take additional arguments. The
arguments are passed in the order they are documented.

Args:
    converter (Callable): A callable that converts the passed value.

    takes_self (bool):
        Pass the partially initialized instance that is being initialized
        as a positional argument. (default: `False`)

    takes_field (bool):
        Pass the field definition (an :class:`Attribute`) into the
        converter as a positional argument. (default: `False`)

.. versionadded:: 24.1.0

### Function: make_class(name, attrs, bases, class_body)

**Description:** A quick way to create a new class called *name* with *attrs*.

.. note::

    ``make_class()`` is a thin wrapper around `attr.s`, not `attrs.define`
    which means that it doesn't come with some of the improved defaults.

    For example, if you want the same ``on_setattr`` behavior as in
    `attrs.define`, you have to pass the hooks yourself: ``make_class(...,
    on_setattr=setters.pipe(setters.convert, setters.validate)``

.. warning::

    It is *your* duty to ensure that the class name and the attribute names
    are valid identifiers. ``make_class()`` will *not* validate them for
    you.

Args:
    name (str): The name for the new class.

    attrs (list | dict):
        A list of names or a dictionary of mappings of names to `attr.ib`\
        s / `attrs.field`\ s.

        The order is deduced from the order of the names or attributes
        inside *attrs*.  Otherwise the order of the definition of the
        attributes is used.

    bases (tuple[type, ...]): Classes that the new class will subclass.

    class_body (dict):
        An optional dictionary of class attributes for the new class.

    attributes_arguments: Passed unmodified to `attr.s`.

Returns:
    type: A new class with *attrs*.

.. versionadded:: 17.1.0 *bases*
.. versionchanged:: 18.1.0 If *attrs* is ordered, the order is retained.
.. versionchanged:: 23.2.0 *class_body*
.. versionchanged:: 25.2.0 Class names can now be unicode.

## Class: _AndValidator

**Description:** Compose many validators to a single one.

### Function: and_()

**Description:** A validator that composes multiple validators into one.

When called on a value, it runs all wrapped validators.

Args:
    validators (~collections.abc.Iterable[typing.Callable]):
        Arbitrary number of validators.

.. versionadded:: 17.1.0

### Function: pipe()

**Description:** A converter that composes multiple converters into one.

When called on a value, it runs all wrapped converters, returning the
*last* value.

Type annotations will be inferred from the wrapped converters', if they
have any.

    converters (~collections.abc.Iterable[typing.Callable]):
        Arbitrary number of converters.

.. versionadded:: 20.1.0

### Function: __repr__(self)

### Function: __bool__(self)

### Function: __reduce__(self, _none_constructor, _args)

### Function: __init__(self, cls, these, auto_attribs, props, has_custom_setattr)

### Function: __repr__(self)

### Function: _eval_snippets(self)

**Description:** Evaluate any registered snippets in one go.

### Function: build_class(self)

**Description:** Finalize class based on the accumulated configuration.

Builder cannot be used after calling this method.

### Function: _patch_original_class(self)

**Description:** Apply accumulated methods and return the class.

### Function: _create_slots_class(self)

**Description:** Build and return a new class with a `__slots__` attribute.

### Function: add_repr(self, ns)

### Function: add_str(self)

### Function: _make_getstate_setstate(self)

**Description:** Create custom __setstate__ and __getstate__ methods.

### Function: make_unhashable(self)

### Function: add_hash(self)

### Function: add_init(self)

### Function: add_replace(self)

### Function: add_match_args(self)

### Function: add_attrs_init(self)

### Function: add_eq(self)

### Function: add_order(self)

### Function: add_setattr(self)

### Function: _add_method_dunders_unsafe(self, method)

**Description:** Add __module__ and __qualname__ to a *method*.

### Function: _add_method_dunders_safe(self, method)

**Description:** Add __module__ and __qualname__ to a *method* if possible.

### Function: decide_callable_or_boolean(value)

**Description:** Decide whether a key function is used.

### Function: wrap(cls)

### Function: append_hash_computation_lines(prefix, indent)

**Description:** Generate the code for actually computing the hash code.
Below this will either be returned directly or used to compute
a value which is then cached, depending on the value of cache_hash

### Function: attrs_to_tuple(obj)

**Description:** Save us some typing.

### Function: __lt__(self, other)

**Description:** Automatically created by attrs.

### Function: __le__(self, other)

**Description:** Automatically created by attrs.

### Function: __gt__(self, other)

**Description:** Automatically created by attrs.

### Function: __ge__(self, other)

**Description:** Automatically created by attrs.

### Function: __init__(self, name, default, validator, repr, cmp, hash, init, inherited, metadata, type, converter, kw_only, eq, eq_key, order, order_key, on_setattr, alias)

### Function: __setattr__(self, name, value)

### Function: from_counting_attr(cls, name, ca, kw_only, type)

### Function: evolve(self)

**Description:** Copy *self* and apply *changes*.

This works similarly to `attrs.evolve` but that function does not work
with :class:`attrs.Attribute`.

It is mainly meant to be used for `transform-fields`.

.. versionadded:: 20.3.0

### Function: __getstate__(self)

**Description:** Play nice with pickle.

### Function: __setstate__(self, state)

**Description:** Play nice with pickle.

### Function: _setattrs(self, name_values_pairs)

### Function: __init__(self, default, validator, repr, cmp, hash, init, converter, metadata, type, kw_only, eq, eq_key, order, order_key, on_setattr, alias)

### Function: validator(self, meth)

**Description:** Decorator that adds *meth* to the list of validators.

Returns *meth* unchanged.

.. versionadded:: 17.1.0

### Function: default(self, meth)

**Description:** Decorator that allows to set the default for an attribute.

Returns *meth* unchanged.

Raises:
    DefaultAlreadySetError: If default has been set before.

.. versionadded:: 17.1.0

## Class: Hashability

**Description:** The hashability of a class.

.. versionadded:: 25.4.0

## Class: KeywordOnly

**Description:** How attributes should be treated regarding keyword-only parameters.

.. versionadded:: 25.4.0

### Function: __init__(self, is_exception, is_slotted, has_weakref_slot, is_frozen, kw_only, collected_fields_by_mro, added_init, added_repr, added_eq, added_ordering, hashability, added_match_args, added_str, added_pickling, on_setattr_hook, field_transformer)

### Function: is_hashable(self)

### Function: __init__(self, factory, takes_self)

### Function: __getstate__(self)

**Description:** Play nice with pickle.

### Function: __setstate__(self, state)

**Description:** Play nice with pickle.

### Function: __init__(self, converter)

### Function: _get_global_name(attr_name)

**Description:** Return the name that a converter for an attribute name *attr_name*
would have.

### Function: _fmt_converter_call(self, attr_name, value_var)

**Description:** Return a string that calls the converter for an attribute name
*attr_name* and the value in variable named *value_var* according to
`self.takes_self` and `self.takes_field`.

### Function: __getstate__(self)

**Description:** Return a dict containing only converter and takes_self -- the rest gets
computed when loading.

### Function: __setstate__(self, state)

**Description:** Load instance from state.

### Function: __call__(self, inst, attr, value)

### Function: getter(self, i)

### Function: _attach_repr(cls_dict, globs)

### Function: __str__(self)

### Function: slots_getstate(self)

**Description:** Automatically created by attrs.

### Function: slots_setstate(self, state)

**Description:** Automatically created by attrs.

### Function: attach_hash(cls_dict, locs)

### Function: _attach_init(cls_dict, globs)

### Function: _attach_attrs_init(cls_dict, globs)

### Function: _attach_eq(cls_dict, globs)

### Function: __setattr__(self, name, val)

### Function: fmt_setter(attr_name, value_var, has_on_setattr)

### Function: fmt_setter_with_converter(attr_name, value_var, has_on_setattr, converter)

### Function: pipe_converter(val, inst, field)

### Function: pipe_converter(val)
