## AI Summary

A file named oinspect.py.


## Class: OInfo

### Function: pylight(code)

## Class: InfoDict

### Function: __getattr__(name)

## Class: InspectorHookData

**Description:** Data passed to the mime hook

### Function: object_info()

**Description:** Make an object info dict with all fields present.

### Function: get_encoding(obj)

**Description:** Get encoding for python source file defining obj

Returns None if obj is not defined in a sourcefile.

### Function: getdoc(obj)

**Description:** Stable wrapper around inspect.getdoc.

This can't crash because of attribute problems.

It also attempts to call a getdoc() method on the given object.  This
allows objects which provide their docstrings via non-standard mechanisms
(like Pyro proxies) to still be inspected by ipython's ? system.

### Function: getsource(obj, oname)

**Description:** Wrapper around inspect.getsource.

This can be modified by other projects to provide customized source
extraction.

Parameters
----------
obj : object
    an object whose source code we will attempt to extract
oname : str
    (optional) a name under which the object is known

Returns
-------
src : unicode or None

### Function: is_simple_callable(obj)

**Description:** True if obj is a function ()

### Function: getargspec(obj)

**Description:** Wrapper around :func:`inspect.getfullargspec`

In addition to functions and methods, this can also handle objects with a
``__call__`` attribute.

DEPRECATED: Deprecated since 7.10. Do not use, will be removed.

### Function: format_argspec(argspec)

**Description:** Format argspect, convenience wrapper around inspect's.

This takes a dict instead of ordered arguments and calls
inspect.format_argspec with the arguments in the necessary order.

DEPRECATED (since 7.10): Do not use; will be removed in future versions.

### Function: call_tip(oinfo, format_call)

**Description:** DEPRECATED since 6.0. Extract call tip data from an oinfo dict.

### Function: _get_wrapped(obj)

**Description:** Get the original object if wrapped in one or more @decorators

Some objects automatically construct similar objects on any unrecognised
attribute access (e.g. unittest.mock.call). To protect against infinite loops,
this will arbitrarily cut off after 100 levels of obj.__wrapped__
attribute access. --TK, Jan 2016

### Function: find_file(obj)

**Description:** Find the absolute path to the file where an object was defined.

This is essentially a robust wrapper around `inspect.getabsfile`.

Returns None if no file can be found.

Parameters
----------
obj : any Python object

Returns
-------
fname : str
    The absolute path to the file where the object was defined.

### Function: find_source_lines(obj)

**Description:** Find the line number in a file where an object was defined.

This is essentially a robust wrapper around `inspect.getsourcelines`.

Returns None if no file can be found.

Parameters
----------
obj : any Python object

Returns
-------
lineno : int
    The line number where the object definition starts.

## Class: Inspector

### Function: _render_signature(obj_signature, obj_name)

**Description:** This was mostly taken from inspect.Signature.__str__.
Look there for the comments.
The only change is to add linebreaks when this gets too long.

### Function: get(self, field)

**Description:** Get a field from the object for backward compatibility with before 8.12

see https://github.com/h5py/h5py/issues/2253

### Function: __init__(self, color_table, code_color_table, scheme, str_detail_level, parent, config)

### Function: _getdef(self, obj, oname)

**Description:** Return the call signature for any callable object.

If any exception is generated, None is returned instead and the
exception is suppressed.

### Function: __head(self, h)

**Description:** Return a header string with proper colors.

### Function: set_active_scheme(self, scheme)

### Function: noinfo(self, msg, oname)

**Description:** Generic message when no information is found.

### Function: pdef(self, obj, oname)

**Description:** Print the call signature for any callable object.

If the object is a class, print the constructor information.

### Function: pdoc(self, obj, oname, formatter)

**Description:** Print the docstring for any object.

Optional:
-formatter: a function to run the docstring through for specially
formatted docstrings.

Examples
--------
In [1]: class NoInit:
   ...:     pass

In [2]: class NoDoc:
   ...:     def __init__(self):
   ...:         pass

In [3]: %pdoc NoDoc
No documentation found for NoDoc

In [4]: %pdoc NoInit
No documentation found for NoInit

In [5]: obj = NoInit()

In [6]: %pdoc obj
No documentation found for obj

In [5]: obj2 = NoDoc()

In [6]: %pdoc obj2
No documentation found for obj2

### Function: psource(self, obj, oname)

**Description:** Print the source code for an object.

### Function: pfile(self, obj, oname)

**Description:** Show the whole file where an object was defined.

### Function: _mime_format(self, text, formatter)

**Description:** Return a mime bundle representation of the input text.

- if `formatter` is None, the returned mime bundle has
   a ``text/plain`` field, with the input text.
   a ``text/html`` field with a ``<pre>`` tag containing the input text.

- if ``formatter`` is not None, it must be a callable transforming the
  input text into a mime bundle. Default values for ``text/plain`` and
  ``text/html`` representations are the ones described above.

Note:

Formatters returning strings are supported but this behavior is deprecated.

### Function: format_mime(self, bundle)

**Description:** Format a mimebundle being created by _make_info_unformatted into a real mimebundle

### Function: _append_info_field(self, bundle, title, key, info, omit_sections, formatter)

**Description:** Append an info value to the unformatted mimebundle being constructed by _make_info_unformatted

### Function: _make_info_unformatted(self, obj, info, formatter, detail_level, omit_sections)

**Description:** Assemble the mimebundle as unformatted lists of information

### Function: _get_info(self, obj, oname, formatter, info, detail_level, omit_sections)

**Description:** Retrieve an info dict and format it.

Parameters
----------
obj : any
    Object to inspect and return info from
oname : str (default: ''):
    Name of the variable pointing to `obj`.
formatter : callable
info
    already computed information
detail_level : integer
    Granularity of detail level, if set to 1, give more information.
omit_sections : list[str]
    Titles or keys to omit from output (can be set, tuple, etc., anything supporting `in`)

### Function: pinfo(self, obj, oname, formatter, info, detail_level, enable_html_pager, omit_sections)

**Description:** Show detailed information about an object.

Optional arguments:

- oname: name of the variable pointing to the object.

- formatter: callable (optional)
      A special formatter for docstrings.

      The formatter is a callable that takes a string as an input
      and returns either a formatted string or a mime type bundle
      in the form of a dictionary.

      Although the support of custom formatter returning a string
      instead of a mime type bundle is deprecated.

- info: a structure with some information fields which may have been
  precomputed already.

- detail_level: if set to 1, more information is given.

- omit_sections: set of section keys and titles to omit

### Function: _info(self, obj, oname, info, detail_level)

**Description:** Inspector.info() was likely improperly marked as deprecated
while only a parameter was deprecated. We "un-deprecate" it.

### Function: info(self, obj, oname, info, detail_level)

**Description:** Compute a dict with detailed information about an object.

Parameters
----------
obj : any
    An object to find information about
oname : str (default: '')
    Name of the variable pointing to `obj`.
info : (default: None)
    A struct (dict like with attr access) with some information fields
    which may have been precomputed already.
detail_level : int (default:0)
    If set to 1, more information is given.

Returns
-------
An object info dict with known fields from `info_fields` (see `InfoDict`).

### Function: _source_contains_docstring(src, doc)

**Description:** Check whether the source *src* contains the docstring *doc*.

This is is helper function to skip displaying the docstring if the
source already contains it, avoiding repetition of information.

### Function: psearch(self, pattern, ns_table, ns_search, ignore_case, show_all)

**Description:** Search namespaces with wildcards for objects.

Arguments:

- pattern: string containing shell-like wildcards to use in namespace
  searches and optionally a type specification to narrow the search to
  objects of that type.

- ns_table: dict of name->namespaces for search.

Optional arguments:

  - ns_search: list of namespace names to include in search.

  - ignore_case(False): make the search case-insensitive.

  - show_all(False): show all names, including those starting with
    underscores.

  - list_types(False): list all available object types for object matching.

### Function: append_field(bundle, title, key, formatter)

### Function: code_formatter(text)
