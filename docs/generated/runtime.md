## AI Summary

A file named runtime.py.


### Function: identity(x)

**Description:** Returns its argument. Useful for certain things in the
environment.

### Function: markup_join(seq)

**Description:** Concatenation that escapes if necessary and converts to string.

### Function: str_join(seq)

**Description:** Simple args to string conversion and concatenation.

### Function: new_context(environment, template_name, blocks, vars, shared, globals, locals)

**Description:** Internal helper for context creation.

## Class: TemplateReference

**Description:** The `self` in templates.

### Function: _dict_method_all(dict_method)

## Class: Context

**Description:** The template context holds the variables of a template.  It stores the
values passed to the template and also the names the template exports.
Creating instances is neither supported nor useful as it's created
automatically at various stages of the template evaluation and should not
be created by hand.

The context is immutable.  Modifications on :attr:`parent` **must not**
happen and modifications on :attr:`vars` are allowed from generated
template code only.  Template filters and global functions marked as
:func:`pass_context` get the active context passed as first argument
and are allowed to access the context read-only.

The template context supports read only dict operations (`get`,
`keys`, `values`, `items`, `iterkeys`, `itervalues`, `iteritems`,
`__getitem__`, `__contains__`).  Additionally there is a :meth:`resolve`
method that doesn't fail with a `KeyError` but returns an
:class:`Undefined` object for missing variables.

## Class: BlockReference

**Description:** One block on a template reference.

## Class: LoopContext

**Description:** A wrapper iterable for dynamic ``for`` loops, with information
about the loop and iteration.

## Class: AsyncLoopContext

## Class: Macro

**Description:** Wraps a macro function.

## Class: Undefined

**Description:** The default undefined type. This can be printed, iterated, and treated as
a boolean. Any other operation will raise an :exc:`UndefinedError`.

>>> foo = Undefined(name='foo')
>>> str(foo)
''
>>> not foo
True
>>> foo + 42
Traceback (most recent call last):
  ...
jinja2.exceptions.UndefinedError: 'foo' is undefined

### Function: make_logging_undefined(logger, base)

**Description:** Given a logger object this returns a new undefined class that will
log certain failures.  It will log iterations and printing.  If no
logger is given a default logger is created.

Example::

    logger = logging.getLogger(__name__)
    LoggingUndefined = make_logging_undefined(
        logger=logger,
        base=Undefined
    )

.. versionadded:: 2.8

:param logger: the logger to use.  If not provided, a default logger
               is created.
:param base: the base class to add logging functionality to.  This
             defaults to :class:`Undefined`.

## Class: ChainableUndefined

**Description:** An undefined that is chainable, where both ``__getattr__`` and
``__getitem__`` return itself rather than raising an
:exc:`UndefinedError`.

>>> foo = ChainableUndefined(name='foo')
>>> str(foo.bar['baz'])
''
>>> foo.bar['baz'] + 42
Traceback (most recent call last):
  ...
jinja2.exceptions.UndefinedError: 'foo' is undefined

.. versionadded:: 2.11.0

## Class: DebugUndefined

**Description:** An undefined that returns the debug info when printed.

>>> foo = DebugUndefined(name='foo')
>>> str(foo)
'{{ foo }}'
>>> not foo
True
>>> foo + 42
Traceback (most recent call last):
  ...
jinja2.exceptions.UndefinedError: 'foo' is undefined

## Class: StrictUndefined

**Description:** An undefined that barks on print and iteration as well as boolean
tests and all kinds of comparisons.  In other words: you can do nothing
with it except checking if it's defined using the `defined` test.

>>> foo = StrictUndefined(name='foo')
>>> str(foo)
Traceback (most recent call last):
  ...
jinja2.exceptions.UndefinedError: 'foo' is undefined
>>> not foo
Traceback (most recent call last):
  ...
jinja2.exceptions.UndefinedError: 'foo' is undefined
>>> foo + 42
Traceback (most recent call last):
  ...
jinja2.exceptions.UndefinedError: 'foo' is undefined

## Class: LoopRenderFunc

### Function: __init__(self, context)

### Function: __getitem__(self, name)

### Function: __repr__(self)

### Function: f_all(self)

### Function: __init__(self, environment, parent, name, blocks, globals)

### Function: super(self, name, current)

**Description:** Render a parent block.

### Function: get(self, key, default)

**Description:** Look up a variable by name, or return a default if the key is
not found.

:param key: The variable name to look up.
:param default: The value to return if the key is not found.

### Function: resolve(self, key)

**Description:** Look up a variable by name, or return an :class:`Undefined`
object if the key is not found.

If you need to add custom behavior, override
:meth:`resolve_or_missing`, not this method. The various lookup
functions use that method, not this one.

:param key: The variable name to look up.

### Function: resolve_or_missing(self, key)

**Description:** Look up a variable by name, or return a ``missing`` sentinel
if the key is not found.

Override this method to add custom lookup behavior.
:meth:`resolve`, :meth:`get`, and :meth:`__getitem__` use this
method. Don't call this method directly.

:param key: The variable name to look up.

### Function: get_exported(self)

**Description:** Get a new dict with the exported variables.

### Function: get_all(self)

**Description:** Return the complete context as dict including the exported
variables.  For optimizations reasons this might not return an
actual copy so be careful with using it.

### Function: call(__self, __obj)

**Description:** Call the callable with the arguments and keyword arguments
provided but inject the active context or environment as first
argument if the callable has :func:`pass_context` or
:func:`pass_environment`.

### Function: derived(self, locals)

**Description:** Internal helper function to create a derived context.  This is
used in situations where the system needs a new context in the same
template that is independent.

### Function: __contains__(self, name)

### Function: __getitem__(self, key)

**Description:** Look up a variable by name with ``[]`` syntax, or raise a
``KeyError`` if the key is not found.

### Function: __repr__(self)

### Function: __init__(self, name, context, stack, depth)

### Function: super(self)

**Description:** Super the block.

### Function: __call__(self)

### Function: __init__(self, iterable, undefined, recurse, depth0)

**Description:** :param iterable: Iterable to wrap.
:param undefined: :class:`Undefined` class to use for next and
    previous items.
:param recurse: The function to render the loop body when the
    loop is marked recursive.
:param depth0: Incremented when looping recursively.

### Function: _to_iterator(iterable)

### Function: length(self)

**Description:** Length of the iterable.

If the iterable is a generator or otherwise does not have a
size, it is eagerly evaluated to get a size.

### Function: __len__(self)

### Function: depth(self)

**Description:** How many levels deep a recursive loop currently is, starting at 1.

### Function: index(self)

**Description:** Current iteration of the loop, starting at 1.

### Function: revindex0(self)

**Description:** Number of iterations from the end of the loop, ending at 0.

Requires calculating :attr:`length`.

### Function: revindex(self)

**Description:** Number of iterations from the end of the loop, ending at 1.

Requires calculating :attr:`length`.

### Function: first(self)

**Description:** Whether this is the first iteration of the loop.

### Function: _peek_next(self)

**Description:** Return the next element in the iterable, or :data:`missing`
if the iterable is exhausted. Only peeks one item ahead, caching
the result in :attr:`_last` for use in subsequent checks. The
cache is reset when :meth:`__next__` is called.

### Function: last(self)

**Description:** Whether this is the last iteration of the loop.

Causes the iterable to advance early. See
:func:`itertools.groupby` for issues this can cause.
The :func:`groupby` filter avoids that issue.

### Function: previtem(self)

**Description:** The item in the previous iteration. Undefined during the
first iteration.

### Function: nextitem(self)

**Description:** The item in the next iteration. Undefined during the last
iteration.

Causes the iterable to advance early. See
:func:`itertools.groupby` for issues this can cause.
The :func:`jinja-filters.groupby` filter avoids that issue.

### Function: cycle(self)

**Description:** Return a value from the given args, cycling through based on
the current :attr:`index0`.

:param args: One or more values to cycle through.

### Function: changed(self)

**Description:** Return ``True`` if previously called with a different value
(including when called for the first time).

:param value: One or more values to compare to the last call.

### Function: __iter__(self)

### Function: __next__(self)

### Function: __call__(self, iterable)

**Description:** When iterating over nested data, render the body of the loop
recursively with the given inner iterable data.

The loop must have the ``recursive`` marker for this to work.

### Function: __repr__(self)

### Function: _to_iterator(iterable)

### Function: __aiter__(self)

### Function: __init__(self, environment, func, name, arguments, catch_kwargs, catch_varargs, caller, default_autoescape)

### Function: __call__(self)

### Function: _invoke(self, arguments, autoescape)

### Function: __repr__(self)

### Function: __init__(self, hint, obj, name, exc)

### Function: _undefined_message(self)

**Description:** Build a message about the undefined value based on how it was
accessed.

### Function: _fail_with_undefined_error(self)

**Description:** Raise an :exc:`UndefinedError` when operations are performed
on the undefined value.

### Function: __getattr__(self, name)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __hash__(self)

### Function: __str__(self)

### Function: __len__(self)

### Function: __iter__(self)

### Function: __bool__(self)

### Function: __repr__(self)

### Function: _log_message(undef)

## Class: LoggingUndefined

### Function: __html__(self)

### Function: __getattr__(self, name)

### Function: __getitem__(self, _name)

### Function: __str__(self)

### Function: __call__(self, reciter, loop_render_func, depth)

### Function: _fail_with_undefined_error(self)

### Function: __str__(self)

### Function: __iter__(self)

### Function: __bool__(self)
