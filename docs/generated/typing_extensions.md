## AI Summary

A file named typing_extensions.py.


## Class: _Sentinel

## Class: _SpecialForm

## Class: _ExtensionsSpecialForm

### Function: IntVar(name)

### Function: _get_protocol_attrs(cls)

### Function: _caller(depth, default)

### Function: _set_default(type_param, default)

### Function: _set_module(typevarlike)

## Class: _DefaultMixin

**Description:** Mixin for TypeVarLike defaults.

## Class: _TypeVarLikeMeta

## Class: _EllipsisDummy

### Function: _create_concatenate_alias(origin, parameters)

### Function: _concatenate_getitem(self, parameters)

### Function: _unpack_args()

### Function: _has_generic_or_protocol_as_origin()

### Function: _is_unpacked_typevartuple(x)

## Class: Sentinel

**Description:** Create a unique sentinel object.

*name* should be the name of the variable to which the return value shall be assigned.

*repr*, if supplied, will be used for the repr of the sentinel object.
If not provided, "<name>" will be used.

### Function: __repr__(self)

### Function: _should_collect_from_parameters(t)

### Function: _should_collect_from_parameters(t)

## Class: _AnyMeta

## Class: Any

**Description:** Special type indicating an unconstrained type.
- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.
Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### Function: __init__(self, getitem)

### Function: __getattr__(self, item)

### Function: __mro_entries__(self, bases)

### Function: __repr__(self)

### Function: __reduce__(self)

### Function: __call__(self)

### Function: __or__(self, other)

### Function: __ror__(self, other)

### Function: __instancecheck__(self, obj)

### Function: __subclasscheck__(self, cls)

### Function: __getitem__(self, parameters)

### Function: __repr__(self)

### Function: final(f)

**Description:** This decorator can be used to indicate to type checkers that
the decorated method cannot be overridden, and decorated class
cannot be subclassed. For example:

    class Base:
        @final
        def done(self) -> None:
            ...
    class Sub(Base):
        def done(self) -> None:  # Error reported by type checker
            ...
    @final
    class Leaf:
        ...
    class Other(Leaf):  # Error reported by type checker
        ...

There is no runtime checking of these properties. The decorator
sets the ``__final__`` attribute to ``True`` on the decorated object
to allow runtime introspection.

### Function: disjoint_base(cls)

**Description:** This decorator marks a class as a disjoint base.

Child classes of a disjoint base cannot inherit from other disjoint bases that are
not parent classes of the disjoint base.

For example:

    @disjoint_base
    class Disjoint1: pass

    @disjoint_base
    class Disjoint2: pass

    class Disjoint3(Disjoint1, Disjoint2): pass  # Type checker error

Type checkers can use knowledge of disjoint bases to detect unreachable code
and determine when two types can overlap.

See PEP 800.

### Function: _flatten_literal_params(parameters)

**Description:** An internal helper for Literal creation: flatten Literals among parameters

### Function: _value_and_type_iter(params)

## Class: _LiteralGenericAlias

## Class: _LiteralForm

### Function: overload(func)

**Description:** Decorator for overloaded functions/methods.

In a stub file, place two or more stub definitions for the same
function in a row, each decorated with @overload.  For example:

@overload
def utf8(value: None) -> None: ...
@overload
def utf8(value: bytes) -> bytes: ...
@overload
def utf8(value: str) -> bytes: ...

In a non-stub file (i.e. a regular .py file), do the same but
follow it with an implementation.  The implementation should *not*
be decorated with @overload.  For example:

@overload
def utf8(value: None) -> None: ...
@overload
def utf8(value: bytes) -> bytes: ...
@overload
def utf8(value: str) -> bytes: ...
def utf8(value):
    # implementation goes here

The overloads for a function can be retrieved at runtime using the
get_overloads() function.

### Function: get_overloads(func)

**Description:** Return all defined overloads for *func* as a sequence.

### Function: clear_overloads()

**Description:** Clear all overloads in the registry.

### Function: _is_dunder(attr)

## Class: _SpecialGenericAlias

### Function: _allow_reckless_class_checks(depth)

**Description:** Allow instance and class checks for special stdlib modules.
The abc and functools modules indiscriminately call isinstance() and
issubclass() on the whole MRO of a user class, which may contain protocols.

### Function: _no_init(self)

### Function: _type_check_issubclass_arg_1(arg)

**Description:** Raise TypeError if `arg` is not an instance of `type`
in `issubclass(arg, <protocol>)`.

In most cases, this is verified by type.__subclasscheck__.
Checking it again unnecessarily would slow down issubclass() checks,
so, we don't perform this check unless we absolutely have to.

For various error paths, however,
we want to ensure that *this* error message is shown to the user
where relevant, rather than a typing.py-specific error message.

## Class: _ProtocolMeta

### Function: _proto_hook(cls, other)

## Class: Protocol

### Function: runtime_checkable(cls)

**Description:** Mark a protocol class as a runtime protocol.

Such protocol can be used with isinstance() and issubclass().
Raise TypeError if applied to a non-protocol class.
This allows a simple-minded structural check very similar to
one trick ponies in collections.abc such as Iterable.

For example::

    @runtime_checkable
    class Closable(Protocol):
        def close(self): ...

    assert isinstance(open('/some/file'), Closable)

Warning: this will check only the presence of the required methods,
not their type signatures!

## Class: SupportsInt

**Description:** An ABC with one abstract method __int__.

## Class: SupportsFloat

**Description:** An ABC with one abstract method __float__.

## Class: SupportsComplex

**Description:** An ABC with one abstract method __complex__.

## Class: SupportsBytes

**Description:** An ABC with one abstract method __bytes__.

## Class: SupportsIndex

## Class: SupportsAbs

**Description:** An ABC with one abstract method __abs__ that is covariant in its return type.

## Class: SupportsRound

**Description:** An ABC with one abstract method __round__ that is covariant in its return type.

## Class: Reader

**Description:** Protocol for simple I/O reader instances.

This protocol only supports blocking I/O.

## Class: Writer

**Description:** Protocol for simple I/O writer instances.

This protocol only supports blocking I/O.

## Class: SingletonMeta

## Class: NoDefaultType

**Description:** The type of the NoDefault singleton.

## Class: NoExtraItemsType

**Description:** The type of the NoExtraItems singleton.

### Function: _get_typeddict_qualifiers(annotation_type)

## Class: _TypedDictMeta

### Function: _create_typeddict()

## Class: _TypedDictSpecialForm

### Function: TypedDict(self, args)

**Description:** A simple typed namespace. At runtime it is equivalent to a plain dict.

TypedDict creates a dictionary type such that a type checker will expect all
instances to have a certain set of keys, where each key is
associated with a value of a consistent type. This expectation
is not checked at runtime.

Usage::

    class Point2D(TypedDict):
        x: int
        y: int
        label: str

    a: Point2D = {'x': 1, 'y': 2, 'label': 'good'}  # OK
    b: Point2D = {'z': 3, 'label': 'bad'}           # Fails type check

    assert Point2D(x=1, y=2, label='first') == dict(x=1, y=2, label='first')

The type info can be accessed via the Point2D.__annotations__ dict, and
the Point2D.__required_keys__ and Point2D.__optional_keys__ frozensets.
TypedDict supports an additional equivalent form::

    Point2D = TypedDict('Point2D', {'x': int, 'y': int, 'label': str})

By default, all keys must be present in a TypedDict. It is possible
to override this by specifying totality::

    class Point2D(TypedDict, total=False):
        x: int
        y: int

This means that a Point2D TypedDict can have any of the keys omitted. A type
checker is only expected to support a literal False or True as the value of
the total argument. True is the default, and makes all items defined in the
class body be required.

The Required and NotRequired special forms can also be used to mark
individual keys as being required or not required::

    class Point2D(TypedDict):
        x: int  # the "x" key must always be present (Required is the default)
        y: NotRequired[int]  # the "y" key can be omitted

See PEP 655 for more details on Required and NotRequired.

### Function: is_typeddict(tp)

**Description:** Check if an annotation is a TypedDict class

For example::
    class Film(TypedDict):
        title: str
        year: int

    is_typeddict(Film)  # => True
    is_typeddict(Union[list, str])  # => False

### Function: assert_type()

**Description:** Assert (to the type checker) that the value is of the given type.

When the type checker encounters a call to assert_type(), it
emits an error if the value is not of the specified type::

    def greet(name: str) -> None:
        assert_type(name, str)  # ok
        assert_type(name, int)  # type checker error

At runtime this returns the first argument unchanged and otherwise
does nothing.

### Function: _strip_extras(t)

**Description:** Strips Annotated, Required and NotRequired from a given type.

### Function: get_type_hints(obj, globalns, localns, include_extras)

**Description:** Return type hints for an object.

This is often the same as obj.__annotations__, but it handles
forward references encoded as string literals, adds Optional[t] if a
default value equal to None is set and recursively replaces all
'Annotated[T, ...]', 'Required[T]' or 'NotRequired[T]' with 'T'
(unless 'include_extras=True').

The argument may be a module, class, method, or function. The annotations
are returned as a dictionary. For classes, annotations include also
inherited members.

TypeError is raised if the argument is not of a type that can contain
annotations, and an empty dictionary is returned if no annotations are
present.

BEWARE -- the behavior of globalns and localns is counterintuitive
(unless you are familiar with how eval() and exec() work).  The
search order is locals first, then globals.

- If no dict arguments are passed, an attempt is made to use the
  globals from obj (or the respective module's globals for classes),
  and these are also used as the locals.  If the object does not appear
  to have globals, an empty dictionary is used.

- If one dict argument is passed, it is used for both globals and
  locals.

- If two dict arguments are passed, they specify globals and
  locals, respectively.

### Function: _could_be_inserted_optional(t)

**Description:** detects Union[..., None] pattern

### Function: _clean_optional(obj, hints, globalns, localns)

### Function: get_origin(tp)

**Description:** Get the unsubscripted version of a type.

This supports generic types, Callable, Tuple, Union, Literal, Final, ClassVar
and Annotated. Return None for unsupported types. Examples::

    get_origin(Literal[42]) is Literal
    get_origin(int) is None
    get_origin(ClassVar[int]) is ClassVar
    get_origin(Generic) is Generic
    get_origin(Generic[T]) is Generic
    get_origin(Union[T, int]) is Union
    get_origin(List[Tuple[T, T]][int]) == list
    get_origin(P.args) is P

### Function: get_args(tp)

**Description:** Get type arguments with all substitutions performed.

For unions, basic simplifications used by Union constructor are performed.
Examples::
    get_args(Dict[str, int]) == (str, int)
    get_args(int) == ()
    get_args(Union[int, Union[T, int], str][int]) == (int, str)
    get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
    get_args(Callable[[], T][int]) == ([], int)

### Function: TypeAlias(self, parameters)

**Description:** Special marker indicating that an assignment should
be recognized as a proper type alias definition by type
checkers.

For example::

    Predicate: TypeAlias = Callable[..., bool]

It's invalid when used anywhere except as in the example above.

### Function: __instancecheck__(cls, __instance)

## Class: TypeVar

**Description:** Type variable.

## Class: _Immutable

**Description:** Mixin to indicate that object should not be copied.

## Class: ParamSpecArgs

**Description:** The args for a ParamSpec object.

Given a ParamSpec object P, P.args is an instance of ParamSpecArgs.

ParamSpecArgs objects have a reference back to their ParamSpec:

P.args.__origin__ is P

This type is meant for runtime introspection and has no special meaning to
static type checkers.

## Class: ParamSpecKwargs

**Description:** The kwargs for a ParamSpec object.

Given a ParamSpec object P, P.kwargs is an instance of ParamSpecKwargs.

ParamSpecKwargs objects have a reference back to their ParamSpec:

P.kwargs.__origin__ is P

This type is meant for runtime introspection and has no special meaning to
static type checkers.

## Class: _ConcatenateGenericAlias

### Function: Concatenate(self, parameters)

**Description:** Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a
higher order function which adds, removes or transforms parameters of a
callable.

For example::

   Callable[Concatenate[int, P], int]

See PEP 612 for detailed information.

### Function: TypeGuard(self, parameters)

**Description:** Special typing form used to annotate the return type of a user-defined
type guard function.  ``TypeGuard`` only accepts a single type argument.
At runtime, functions marked this way should return a boolean.

``TypeGuard`` aims to benefit *type narrowing* -- a technique used by static
type checkers to determine a more precise type of an expression within a
program's code flow.  Usually type narrowing is done by analyzing
conditional code flow and applying the narrowing to a block of code.  The
conditional expression here is sometimes referred to as a "type guard".

Sometimes it would be convenient to use a user-defined boolean function
as a type guard.  Such a function should use ``TypeGuard[...]`` as its
return type to alert static type checkers to this intention.

Using  ``-> TypeGuard`` tells the static type checker that for a given
function:

1. The return value is a boolean.
2. If the return value is ``True``, the type of its argument
is the type inside ``TypeGuard``.

For example::

    def is_str(val: Union[str, float]):
        # "isinstance" type guard
        if isinstance(val, str):
            # Type of ``val`` is narrowed to ``str``
            ...
        else:
            # Else, type of ``val`` is narrowed to ``float``.
            ...

Strict type narrowing is not enforced -- ``TypeB`` need not be a narrower
form of ``TypeA`` (it can even be a wider form) and this may lead to
type-unsafe results.  The main reason is to allow for things like
narrowing ``List[object]`` to ``List[str]`` even though the latter is not
a subtype of the former, since ``List`` is invariant.  The responsibility of
writing type-safe type guards is left to the user.

``TypeGuard`` also works with type variables.  For more information, see
PEP 647 (User-Defined Type Guards).

### Function: TypeIs(self, parameters)

**Description:** Special typing form used to annotate the return type of a user-defined
type narrower function.  ``TypeIs`` only accepts a single type argument.
At runtime, functions marked this way should return a boolean.

``TypeIs`` aims to benefit *type narrowing* -- a technique used by static
type checkers to determine a more precise type of an expression within a
program's code flow.  Usually type narrowing is done by analyzing
conditional code flow and applying the narrowing to a block of code.  The
conditional expression here is sometimes referred to as a "type guard".

Sometimes it would be convenient to use a user-defined boolean function
as a type guard.  Such a function should use ``TypeIs[...]`` as its
return type to alert static type checkers to this intention.

Using  ``-> TypeIs`` tells the static type checker that for a given
function:

1. The return value is a boolean.
2. If the return value is ``True``, the type of its argument
is the intersection of the type inside ``TypeIs`` and the argument's
previously known type.

For example::

    def is_awaitable(val: object) -> TypeIs[Awaitable[Any]]:
        return hasattr(val, '__await__')

    def f(val: Union[int, Awaitable[int]]) -> int:
        if is_awaitable(val):
            assert_type(val, Awaitable[int])
        else:
            assert_type(val, int)

``TypeIs`` also works with type variables.  For more information, see
PEP 742 (Narrowing types with TypeIs).

## Class: _TypeFormForm

### Function: TypeForm(self, parameters)

**Description:** A special form representing the value that results from the evaluation
of a type expression. This value encodes the information supplied in the
type expression, and it represents the type described by that type expression.

When used in a type expression, TypeForm describes a set of type form objects.
It accepts a single type argument, which must be a valid type expression.
``TypeForm[T]`` describes the set of all type form objects that represent
the type T or types that are assignable to T.

Usage:

    def cast[T](typ: TypeForm[T], value: Any) -> T: ...

    reveal_type(cast(int, "x"))  # int

See PEP 747 for more information.

### Function: LiteralString(self, params)

**Description:** Represents an arbitrary literal string.

Example::

  from typing_extensions import LiteralString

  def query(sql: LiteralString) -> ...:
      ...

  query("SELECT * FROM table")  # ok
  query(f"SELECT * FROM {input()}")  # not ok

See PEP 675 for details.

### Function: Self(self, params)

**Description:** Used to spell the type of "self" in classes.

Example::

  from typing import Self

  class ReturnsSelf:
      def parse(self, data: bytes) -> Self:
          ...
          return self

### Function: Never(self, params)

**Description:** The bottom type, a type that has no members.

This can be used to define a function that should never be
called, or a function that never returns::

    from typing_extensions import Never

    def never_call_me(arg: Never) -> None:
        pass

    def int_or_str(arg: int | str) -> None:
        never_call_me(arg)  # type checker error
        match arg:
            case int():
                print("It's an int")
            case str():
                print("It's a str")
            case _:
                never_call_me(arg)  # ok, arg is of type Never

### Function: Required(self, parameters)

**Description:** A special typing construct to mark a key of a total=False TypedDict
as required. For example:

    class Movie(TypedDict, total=False):
        title: Required[str]
        year: int

    m = Movie(
        title='The Matrix',  # typechecker error if key is omitted
        year=1999,
    )

There is no runtime checking that a required key is actually provided
when instantiating a related TypedDict.

### Function: NotRequired(self, parameters)

**Description:** A special typing construct to mark a key of a TypedDict as
potentially missing. For example:

    class Movie(TypedDict):
        title: str
        year: NotRequired[int]

    m = Movie(
        title='The Matrix',  # typechecker error if key is omitted
        year=1999,
    )

### Function: ReadOnly(self, parameters)

**Description:** A special typing construct to mark an item of a TypedDict as read-only.

For example:

    class Movie(TypedDict):
        title: ReadOnly[str]
        year: int

    def mutate_movie(m: Movie) -> None:
        m["year"] = 1992  # allowed
        m["title"] = "The Matrix"  # typechecker error

There is no runtime checking for this property.

### Function: _is_unpack(obj)

## Class: _UnpackSpecialForm

## Class: _UnpackAlias

### Function: Unpack(self, parameters)

### Function: _is_unpack(obj)

### Function: reveal_type()

**Description:** Reveal the inferred type of a variable.

When a static type checker encounters a call to ``reveal_type()``,
it will emit the inferred type of the argument::

    x: int = 1
    reveal_type(x)

Running a static type checker (e.g., ``mypy``) on this example
will produce output similar to 'Revealed type is "builtins.int"'.

At runtime, the function prints the runtime type of the
argument and returns it unchanged.

### Function: assert_never()

**Description:** Assert to the type checker that a line of code is unreachable.

Example::

    def int_or_str(arg: int | str) -> None:
        match arg:
            case int():
                print("It's an int")
            case str():
                print("It's a str")
            case _:
                assert_never(arg)

If a type checker finds that a call to assert_never() is
reachable, it will emit an error.

At runtime, this throws an exception when called.

### Function: dataclass_transform()

**Description:** Decorator that marks a function, class, or metaclass as providing
dataclass-like behavior.

Example:

    from typing_extensions import dataclass_transform

    _T = TypeVar("_T")

    # Used on a decorator function
    @dataclass_transform()
    def create_model(cls: type[_T]) -> type[_T]:
        ...
        return cls

    @create_model
    class CustomerModel:
        id: int
        name: str

    # Used on a base class
    @dataclass_transform()
    class ModelBase: ...

    class CustomerModel(ModelBase):
        id: int
        name: str

    # Used on a metaclass
    @dataclass_transform()
    class ModelMeta(type): ...

    class ModelBase(metaclass=ModelMeta): ...

    class CustomerModel(ModelBase):
        id: int
        name: str

Each of the ``CustomerModel`` classes defined in this example will now
behave similarly to a dataclass created with the ``@dataclasses.dataclass``
decorator. For example, the type checker will synthesize an ``__init__``
method.

The arguments to this decorator can be used to customize this behavior:
- ``eq_default`` indicates whether the ``eq`` parameter is assumed to be
  True or False if it is omitted by the caller.
- ``order_default`` indicates whether the ``order`` parameter is
  assumed to be True or False if it is omitted by the caller.
- ``kw_only_default`` indicates whether the ``kw_only`` parameter is
  assumed to be True or False if it is omitted by the caller.
- ``frozen_default`` indicates whether the ``frozen`` parameter is
  assumed to be True or False if it is omitted by the caller.
- ``field_specifiers`` specifies a static list of supported classes
  or functions that describe fields, similar to ``dataclasses.field()``.

At runtime, this decorator records its arguments in the
``__dataclass_transform__`` attribute on the decorated object.

See PEP 681 for details.

### Function: override()

**Description:** Indicate that a method is intended to override a method in a base class.

Usage:

    class Base:
        def method(self) -> None:
            pass

    class Child(Base):
        @override
        def method(self) -> None:
            super().method()

When this decorator is applied to a method, the type checker will
validate that it overrides a method with the same name on a base class.
This helps prevent bugs that may occur when a base class is changed
without an equivalent change to a child class.

There is no runtime checking of these properties. The decorator
sets the ``__override__`` attribute to ``True`` on the decorated object
to allow runtime introspection.

See PEP 698 for details.

## Class: deprecated

**Description:** Indicate that a class, function or overload is deprecated.

When this decorator is applied to an object, the type checker
will generate a diagnostic on usage of the deprecated object.

Usage:

    @deprecated("Use B instead")
    class A:
        pass

    @deprecated("Use g instead")
    def f():
        pass

    @overload
    @deprecated("int support is deprecated")
    def g(x: int) -> int: ...
    @overload
    def g(x: str) -> int: ...

The warning specified by *category* will be emitted at runtime
on use of deprecated objects. For functions, that happens on calls;
for classes, on instantiation and on creation of subclasses.
If the *category* is ``None``, no warning is emitted at runtime.
The *stacklevel* determines where the
warning is emitted. If it is ``1`` (the default), the warning
is emitted at the direct caller of the deprecated object; if it
is higher, it is emitted further up the stack.
Static type checker behavior is not affected by the *category*
and *stacklevel* arguments.

The deprecation message passed to the decorator is saved in the
``__deprecated__`` attribute on the decorated object.
If applied to an overload, the decorator
must be after the ``@overload`` decorator for the attribute to
exist on the overload as returned by ``get_overloads()``.

See PEP 702 for details.

### Function: _is_param_expr(arg)

### Function: _is_param_expr(arg)

### Function: _check_generic(cls, parameters, elen)

**Description:** Check correct count for parameters of a generic cls (internal helper).

This gives a nice error message in case of count mismatch.

### Function: _check_generic(cls, parameters, elen)

**Description:** Check correct count for parameters of a generic cls (internal helper).

This gives a nice error message in case of count mismatch.

### Function: _collect_type_vars(types, typevar_types)

**Description:** Collect all type variable contained in types in order of
first appearance (lexicographic order). For example::

    _collect_type_vars((T, List[S, T])) == (T, S)

### Function: _collect_parameters(args)

**Description:** Collect all type variables and parameter specifications in args
in order of first appearance (lexicographic order).

For example::

    assert _collect_parameters((T, Callable[P, T])) == (T, P)

### Function: _make_nmtuple(name, types, module, defaults)

## Class: _NamedTupleMeta

### Function: _namedtuple_mro_entries(bases)

### Function: NamedTuple()

**Description:** Typed version of namedtuple.

Usage::

    class Employee(NamedTuple):
        name: str
        id: int

This is equivalent to::

    Employee = collections.namedtuple('Employee', ['name', 'id'])

The resulting class has an extra __annotations__ attribute, giving a
dict that maps field names to types.  (The field names are also in
the _fields attribute, which is part of the namedtuple API.)
An alternative equivalent functional syntax is also accepted::

    Employee = NamedTuple('Employee', [('name', str), ('id', int)])

## Class: Buffer

**Description:** Base class for classes that implement the buffer protocol.

The buffer protocol allows Python objects to expose a low-level
memory buffer interface. Before Python 3.12, it is not possible
to implement the buffer protocol in pure Python code, or even
to check whether a class implements the buffer protocol. In
Python 3.12 and higher, the ``__buffer__`` method allows access
to the buffer protocol from Python code, and the
``collections.abc.Buffer`` ABC allows checking whether a class
implements the buffer protocol.

To indicate support for the buffer protocol in earlier versions,
inherit from this ABC, either in a stub file or at runtime,
or use ABC registration. This ABC provides no methods, because
there is no Python-accessible methods shared by pre-3.12 buffer
classes. It is useful primarily for static checks.

### Function: get_original_bases()

**Description:** Return the class's "original" bases prior to modification by `__mro_entries__`.

Examples::

    from typing import TypeVar, Generic
    from typing_extensions import NamedTuple, TypedDict

    T = TypeVar("T")
    class Foo(Generic[T]): ...
    class Bar(Foo[int], float): ...
    class Baz(list[str]): ...
    Eggs = NamedTuple("Eggs", [("a", int), ("b", str)])
    Spam = TypedDict("Spam", {"a": int, "b": str})

    assert get_original_bases(Bar) == (Foo[int], float)
    assert get_original_bases(Baz) == (list[str],)
    assert get_original_bases(Eggs) == (NamedTuple,)
    assert get_original_bases(Spam) == (TypedDict,)
    assert get_original_bases(int) == (object,)

## Class: NewType

**Description:** NewType creates simple unique types with almost zero
runtime overhead. NewType(name, tp) is considered a subtype of tp
by static type checkers. At runtime, NewType(name, tp) returns
a dummy callable that simply returns its argument. Usage::
    UserId = NewType('UserId', int)
    def name_by_id(user_id: UserId) -> str:
        ...
    UserId('user')          # Fails type check
    name_by_id(42)          # Fails type check
    name_by_id(UserId(42))  # OK
    num = UserId(5) + 1     # type: int

## Class: TypeAliasType

**Description:** Create named, parameterized type aliases.

This provides a backport of the new `type` statement in Python 3.12:

    type ListOrSet[T] = list[T] | set[T]

is equivalent to:

    T = TypeVar("T")
    ListOrSet = TypeAliasType("ListOrSet", list[T] | set[T], type_params=(T,))

The name ListOrSet can then be used as an alias for the type it refers to.

The type_params argument should contain all the type parameters used
in the value of the type alias. If the alias is not generic, this
argument is omitted.

Static type checkers should only support type aliases declared using
TypeAliasType that follow these rules:

- The first argument (the name) must be a string literal.
- The TypeAliasType instance must be immediately assigned to a variable
  of the same name. (For example, 'X = TypeAliasType("Y", int)' is invalid,
  as is 'X, Y = TypeAliasType("X", int), TypeAliasType("Y", int)').

### Function: is_protocol()

**Description:** Return True if the given type is a Protocol.

Example::

    >>> from typing_extensions import Protocol, is_protocol
    >>> class P(Protocol):
    ...     def a(self) -> str: ...
    ...     b: int
    >>> is_protocol(P)
    True
    >>> is_protocol(int)
    False

### Function: get_protocol_members()

**Description:** Return the set of members defined in a Protocol.

Example::

    >>> from typing_extensions import Protocol, get_protocol_members
    >>> class P(Protocol):
    ...     def a(self) -> str: ...
    ...     b: int
    >>> get_protocol_members(P)
    frozenset({'a', 'b'})

Raise a TypeError for arguments that are not Protocols.

## Class: Doc

**Description:** Define the documentation of a type annotation using ``Annotated``, to be
 used in class attributes, function and method parameters, return values,
 and variables.

The value should be a positional-only string literal to allow static tools
like editors and documentation generators to use it.

This complements docstrings.

The string value passed is available in the attribute ``documentation``.

Example::

    >>> from typing_extensions import Annotated, Doc
    >>> def hi(to: Annotated[str, Doc("Who to say hi to")]) -> None: ...

## Class: Format

### Function: get_annotations(obj)

**Description:** Compute the annotations dict for an object.

obj may be a callable, class, or module.
Passing in an object of any other type raises TypeError.

Returns a dict.  get_annotations() returns a new dict every time
it's called; calling it twice on the same object will return two
different but equivalent dicts.

This is a backport of `inspect.get_annotations`, which has been
in the standard library since Python 3.10. See the standard library
documentation for more:

    https://docs.python.org/3/library/inspect.html#inspect.get_annotations

This backport adds the *format* argument introduced by PEP 649. The
three formats supported are:
* VALUE: the annotations are returned as-is. This is the default and
  it is compatible with the behavior on previous Python versions.
* FORWARDREF: return annotations as-is if possible, but replace any
  undefined names with ForwardRef objects. The implementation proposed by
  PEP 649 relies on language changes that cannot be backported; the
  typing-extensions implementation simply returns the same result as VALUE.
* STRING: return annotations as strings, in a format close to the original
  source. Again, this behavior cannot be replicated directly in a backport.
  As an approximation, typing-extensions retrieves the annotations under
  VALUE semantics and then stringifies them.

The purpose of this backport is to allow users who would like to use
FORWARDREF or STRING semantics once PEP 649 is implemented, but who also
want to support earlier Python versions, to simply write:

    typing_extensions.get_annotations(obj, format=Format.FORWARDREF)

### Function: _eval_with_owner(forward_ref)

### Function: evaluate_forward_ref(forward_ref)

**Description:** Evaluate a forward reference as a type hint.

This is similar to calling the ForwardRef.evaluate() method,
but unlike that method, evaluate_forward_ref() also:

* Recursively evaluates forward references nested within the type hint.
* Rejects certain objects that are not valid type hints.
* Replaces type hints that evaluate to None with types.NoneType.
* Supports the *FORWARDREF* and *STRING* formats.

*forward_ref* must be an instance of ForwardRef. *owner*, if given,
should be the object that holds the annotations that the forward reference
derived from, such as a module, class object, or function. It is used to
infer the namespaces to use for looking up names. *globals* and *locals*
can also be explicitly given to provide the global and local namespaces.
*type_params* is a tuple of type parameters that are in scope when
evaluating the forward reference. This parameter must be provided (though
it may be an empty tuple) if *owner* is not given and the forward reference
does not already have an owner set. *format* specifies the format of the
annotation and is a member of the annotationlib.Format enum.

### Function: __init__(self, name, repr)

### Function: __repr__(self)

### Function: __getstate__(self)

### Function: type_repr(value)

**Description:** Convert a Python value to a format suitable for use with the STRING format.

This is intended as a helper for tools that support the STRING format but do
not have access to the code that originally produced the annotations. It uses
repr() for most objects.

### Function: __instancecheck__(self, obj)

### Function: __repr__(self)

### Function: __new__(cls)

### Function: __eq__(self, other)

### Function: __hash__(self)

### Function: __init__(self, doc)

### Function: __getitem__(self, parameters)

### Function: __init__(self, origin, nparams)

### Function: __setattr__(self, attr, val)

### Function: __getitem__(self, params)

### Function: __new__(mcls, name, bases, namespace)

### Function: __init__(cls)

### Function: __subclasscheck__(cls, other)

### Function: __instancecheck__(cls, instance)

### Function: __eq__(cls, other)

### Function: __hash__(cls)

### Function: __init_subclass__(cls)

### Function: __int__(self)

### Function: __float__(self)

### Function: __complex__(self)

### Function: __bytes__(self)

### Function: __index__(self)

### Function: __abs__(self)

### Function: __round__(self, ndigits)

### Function: read()

**Description:** Read data from the input stream and return it.

If *size* is specified, at most *size* items (bytes/characters) will be
read.

### Function: write()

**Description:** Write *data* to the output stream and return the number of items written.

### Function: __setattr__(cls, attr, value)

### Function: __new__(cls)

### Function: __repr__(self)

### Function: __reduce__(self)

### Function: __new__(cls)

### Function: __repr__(self)

### Function: __reduce__(self)

### Function: __new__(cls, name, bases, ns)

**Description:** Create new typed dict class object.

This method is called when TypedDict is subclassed,
or when TypedDict is instantiated. This way
TypedDict supports all three syntax forms described in its docstring.
Subclasses and instances of TypedDict return actual dictionaries.

### Function: __subclasscheck__(cls, other)

### Function: __call__()

### Function: __mro_entries__(self, bases)

### Function: __new__(cls, name)

### Function: __init_subclass__(cls)

### Function: __copy__(self)

### Function: __deepcopy__(self, memo)

### Function: __init__(self, origin)

### Function: __repr__(self)

### Function: __eq__(self, other)

### Function: __init__(self, origin)

### Function: __repr__(self)

### Function: __eq__(self, other)

## Class: ParamSpec

**Description:** Parameter specification.

## Class: ParamSpec

**Description:** Parameter specification variable.

Usage::

   P = ParamSpec('P')

Parameter specification variables exist primarily for the benefit of static
type checkers.  They are used to forward the parameter types of one
callable to another callable, a pattern commonly found in higher order
functions and decorators.  They are only valid when used in ``Concatenate``,
or s the first argument to ``Callable``. In Python 3.10 and higher,
they are also supported in user-defined Generics at runtime.
See class Generic for more information on generic types.  An
example for annotating a decorator::

   T = TypeVar('T')
   P = ParamSpec('P')

   def add_logging(f: Callable[P, T]) -> Callable[P, T]:
       '''A type-safe decorator to add logging to a function.'''
       def inner(*args: P.args, **kwargs: P.kwargs) -> T:
           logging.info(f'{f.__name__} was called')
           return f(*args, **kwargs)
       return inner

   @add_logging
   def add_two(x: float, y: float) -> float:
       '''Add two numbers together.'''
       return x + y

Parameter specification variables defined with covariant=True or
contravariant=True can be used to declare covariant or contravariant
generic types.  These keyword arguments are valid, but their actual semantics
are yet to be decided.  See PEP 612 for details.

Parameter specification variables can be introspected. e.g.:

   P.__name__ == 'T'
   P.__bound__ == None
   P.__covariant__ == False
   P.__contravariant__ == False

Note that only parameter specification variables defined in global scope can
be pickled.

### Function: _type_convert(arg, module)

**Description:** For converting None to type(None), and strings to ForwardRef.

### Function: __init__(self, origin, args)

### Function: __repr__(self)

### Function: __hash__(self)

### Function: __call__(self)

### Function: __parameters__(self)

### Function: copy_with(self, params)

### Function: __getitem__(self, args)

## Class: _ConcatenateGenericAlias

### Function: __call__()

### Function: __init__(self, getitem)

### Function: __typing_unpacked_tuple_args__(self)

### Function: __typing_is_unpacked_typevartuple__(self)

### Function: __getitem__(self, args)

## Class: TypeVarTuple

**Description:** Type variable tuple.

## Class: TypeVarTuple

**Description:** Type variable tuple.

Usage::

    Ts = TypeVarTuple('Ts')

In the same way that a normal type variable is a stand-in for a single
type such as ``int``, a type variable *tuple* is a stand-in for a *tuple*
type such as ``Tuple[int, str]``.

Type variable tuples can be used in ``Generic`` declarations.
Consider the following example::

    class Array(Generic[*Ts]): ...

The ``Ts`` type variable tuple here behaves like ``tuple[T1, T2]``,
where ``T1`` and ``T2`` are type variables. To use these type variables
as type parameters of ``Array``, we must *unpack* the type variable tuple using
the star operator: ``*Ts``. The signature of ``Array`` then behaves
as if we had simply written ``class Array(Generic[T1, T2]): ...``.
In contrast to ``Generic[T1, T2]``, however, ``Generic[*Shape]`` allows
us to parameterise the class with an *arbitrary* number of type parameters.

Type variable tuples can be used anywhere a normal ``TypeVar`` can.
This includes class definitions, as shown above, as well as function
signatures and variable annotations::

    class Array(Generic[*Ts]):

        def __init__(self, shape: Tuple[*Ts]):
            self._shape: Tuple[*Ts] = shape

        def get_shape(self) -> Tuple[*Ts]:
            return self._shape

    shape = (Height(480), Width(640))
    x: Array[Height, Width] = Array(shape)
    y = abs(x)  # Inferred type is Array[Height, Width]
    z = x + x   #        ...    is Array[Height, Width]
    x.get_shape()  #     ...    is tuple[Height, Width]

### Function: decorator(cls_or_fn)

### Function: __init__()

### Function: __call__()

### Function: __new__(cls, typename, bases, ns)

### Function: __call__()

### Function: __init__(self, name, tp)

### Function: __mro_entries__(self, bases)

### Function: __repr__(self)

### Function: __reduce__(self)

### Function: _is_unionable(obj)

**Description:** Corresponds to is_unionable() in unionobject.c in CPython.

### Function: _is_unionable(obj)

**Description:** Corresponds to is_unionable() in unionobject.c in CPython.

## Class: _TypeAliasGenericAlias

### Function: __init__(self, name, value)

### Function: __setattr__()

### Function: __delattr__()

### Function: _raise_attribute_error(self, name)

### Function: __repr__(self)

### Function: _check_parameters(self, parameters)

### Function: __getitem__(self, parameters)

### Function: __reduce__(self)

### Function: __init_subclass__(cls)

### Function: __call__(self)

### Function: __init__()

### Function: __repr__(self)

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: __call__(self)

### Function: __or__(self, other)

### Function: __ror__(self, other)

### Function: _tvar_prepare_subst(alias, args)

### Function: __new__(cls, name)

### Function: __init_subclass__(cls)

### Function: args(self)

### Function: kwargs(self)

### Function: __init__(self, name)

### Function: __repr__(self)

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: __reduce__(self)

### Function: __call__(self)

### Function: copy_with(self, params)

### Function: __getitem__(self, args)

### Function: __new__(cls, name)

### Function: __init_subclass__(self)

### Function: __iter__(self)

### Function: __init__(self, name)

### Function: __repr__(self)

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: __reduce__(self)

### Function: __init_subclass__(self)

## Class: Dummy

### Function: __or__(self, other)

### Function: __ror__(self, other)

### Function: __getattr__(self, attr)

### Function: _check_single_param(self, param, recursion)

### Function: __or__(self, right)

### Function: __ror__(self, left)

### Function: __annotate__(format)

### Function: _paramspec_prepare_subst(alias, args)

### Function: _typevartuple_prepare_subst(alias, args)

### Function: __init_subclass__(cls)

### Function: __new__()

### Function: __init_subclass__()

### Function: __init_subclass__()

### Function: wrapper()
