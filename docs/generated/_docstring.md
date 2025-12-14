## AI Summary

A file named _docstring.py.


### Function: kwarg_doc(text)

**Description:** Decorator for defining the kwdoc documentation of artist properties.

This decorator can be applied to artist property setter methods.
The given text is stored in a private attribute ``_kwarg_doc`` on
the method.  It is used to overwrite auto-generated documentation
in the *kwdoc list* for artists. The kwdoc list is used to document
``**kwargs`` when they are properties of an artist. See e.g. the
``**kwargs`` section in `.Axes.text`.

The text should contain the supported types, as well as the default
value if applicable, e.g.:

    @_docstring.kwarg_doc("bool, default: :rc:`text.usetex`")
    def set_usetex(self, usetex):

See Also
--------
matplotlib.artist.kwdoc

## Class: Substitution

**Description:** A decorator that performs %-substitution on an object's docstring.

This decorator should be robust even if ``obj.__doc__`` is None (for
example, if -OO was passed to the interpreter).

Usage: construct a docstring.Substitution with a sequence or dictionary
suitable for performing substitution; then decorate a suitable function
with the constructed object, e.g.::

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

One can also use positional arguments::

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"

## Class: _ArtistKwdocLoader

## Class: _ArtistPropertiesSubstitution

**Description:** A class to substitute formatted placeholders in docstrings.

This is realized in a single instance ``_docstring.interpd``.

Use `~._ArtistPropertiesSubstition.register` to define placeholders and
their substitution, e.g. ``_docstring.interpd.register(name="some value")``.

Use this as a decorator to apply the substitution::

    @_docstring.interpd
    def some_func():
        '''Replace %(name)s.'''

Decorating a class triggers substitution both on the class docstring and
on the class' ``__init__`` docstring (which is a commonly required
pattern for Artist subclasses).

Substitutions of the form ``%(classname:kwdoc)s`` (ending with the
literal ":kwdoc" suffix) trigger lookup of an Artist subclass with the
given *classname*, and are substituted with the `.kwdoc` of that class.

### Function: copy(source)

**Description:** Copy a docstring from another source function (if present).

### Function: decorator(func)

### Function: __init__(self)

### Function: __call__(self, func)

### Function: __missing__(self, key)

### Function: __init__(self)

### Function: register(self)

**Description:** Register substitutions.

``_docstring.interpd.register(name="some value")`` makes "name" available
as a named parameter that will be replaced by "some value".

### Function: __call__(self, obj)

### Function: do_copy(target)
