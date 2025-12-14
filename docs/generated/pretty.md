## AI Summary

A file named pretty.py.


### Function: _safe_getattr(obj, attr, default)

**Description:** Safe version of getattr.

Same as getattr, but will return ``default`` on any Exception,
rather than raising.

## Class: CUnicodeIO

### Function: _sorted_for_pprint(items)

**Description:** Sort the given items for pretty printing. Since some predictable
sorting is better than no sorting at all, we sort on the string
representation if normal sorting fails.

### Function: pretty(obj, verbose, max_width, newline, max_seq_length)

**Description:** Pretty print the object's representation.

### Function: pprint(obj, verbose, max_width, newline, max_seq_length)

**Description:** Like `pretty` but print to stdout.

## Class: _PrettyPrinterBase

## Class: PrettyPrinter

**Description:** Baseclass for the `RepresentationPrinter` prettyprinter that is used to
generate pretty reprs of objects.  Contrary to the `RepresentationPrinter`
this printer knows nothing about the default pprinters or the `_repr_pretty_`
callback method.

### Function: _get_mro(obj_class)

**Description:** Get a reasonable method resolution order of a class and its superclasses
for both old-style and new-style classes.

## Class: RepresentationPrinter

**Description:** Special pretty printer that has a `pretty` method that calls the pretty
printer for a python object.

This class stores processing data on `self` so you must *never* use
this class in a threaded environment.  Always lock it or reinstanciate
it.

Instances also have a verbose flag callbacks can access to control their
output.  For example the default instance repr prints all attributes and
methods that are not prefixed by an underscore if the printer is in
verbose mode.

## Class: Printable

## Class: Text

## Class: Breakable

## Class: Group

## Class: GroupQueue

## Class: RawText

**Description:** Object such that ``p.pretty(RawText(value))`` is the same as ``p.text(value)``.

An example usage of this would be to show a list as binary numbers, using
``p.pretty([RawText(bin(i)) for i in integers])``.

## Class: CallExpression

**Description:** Object which emits a line-wrapped call expression in the form `__name(*args, **kwargs)` 

## Class: RawStringLiteral

**Description:** Wrapper that shows a string with a `r` prefix 

### Function: _default_pprint(obj, p, cycle)

**Description:** The default print function.  Used if an object does not provide one and
it's none of the builtin objects.

### Function: _seq_pprinter_factory(start, end)

**Description:** Factory that returns a pprint function useful for sequences.  Used by
the default pprint for tuples and lists.

### Function: _set_pprinter_factory(start, end)

**Description:** Factory that returns a pprint function useful for sets and frozensets.

### Function: _dict_pprinter_factory(start, end)

**Description:** Factory that returns a pprint function used by the default pprint of
dicts and dict proxies.

### Function: _super_pprint(obj, p, cycle)

**Description:** The pprint for the super type.

## Class: _ReFlags

### Function: _re_pattern_pprint(obj, p, cycle)

**Description:** The pprint function for regular expression patterns.

### Function: _types_simplenamespace_pprint(obj, p, cycle)

**Description:** The pprint function for types.SimpleNamespace.

### Function: _type_pprint(obj, p, cycle)

**Description:** The pprint for classes and types.

### Function: _repr_pprint(obj, p, cycle)

**Description:** A pprint that just redirects to the normal repr function.

### Function: _function_pprint(obj, p, cycle)

**Description:** Base pprint for all functions and builtin functions.

### Function: _exception_pprint(obj, p, cycle)

**Description:** Base pprint for all exceptions.

### Function: for_type(typ, func)

**Description:** Add a pretty printer for a given type.

### Function: for_type_by_name(type_module, type_name, func)

**Description:** Add a pretty printer for a type specified by the module and name of a type
rather than the type object itself.

### Function: _defaultdict_pprint(obj, p, cycle)

### Function: _ordereddict_pprint(obj, p, cycle)

### Function: _deque_pprint(obj, p, cycle)

### Function: _counter_pprint(obj, p, cycle)

### Function: _userlist_pprint(obj, p, cycle)

### Function: __init__(self)

### Function: indent(self, indent)

**Description:** with statement support for indenting/dedenting.

### Function: group(self, indent, open, close)

**Description:** like begin_group / end_group but for the with statement.

### Function: __init__(self, output, max_width, newline, max_seq_length)

### Function: _break_one_group(self, group)

### Function: _break_outer_groups(self)

### Function: text(self, obj)

**Description:** Add literal text to the output.

### Function: breakable(self, sep)

**Description:** Add a breakable separator to the output.  This does not mean that it
will automatically break here.  If no breaking on this position takes
place the `sep` is inserted which default to one space.

### Function: break_(self)

**Description:** Explicitly insert a newline into the output, maintaining correct indentation.

### Function: begin_group(self, indent, open)

**Description:** Begin a group.
The first parameter specifies the indentation for the next line (usually
the width of the opening text), the second the opening text.  All
parameters are optional.

### Function: _enumerate(self, seq)

**Description:** like enumerate, but with an upper limit on the number of items

### Function: end_group(self, dedent, close)

**Description:** End a group. See `begin_group` for more details.

### Function: flush(self)

**Description:** Flush data that is left in the buffer.

### Function: __init__(self, output, verbose, max_width, newline, singleton_pprinters, type_pprinters, deferred_pprinters, max_seq_length)

### Function: pretty(self, obj)

**Description:** Pretty print the given object.

### Function: _in_deferred_types(self, cls)

**Description:** Check if the given class is specified in the deferred type registry.

Returns the printer from the registry if it exists, and None if the
class is not in the registry. Successful matches will be moved to the
regular type registry for future use.

### Function: output(self, stream, output_width)

### Function: __init__(self)

### Function: output(self, stream, output_width)

### Function: add(self, obj, width)

### Function: __init__(self, seq, width, pretty)

### Function: output(self, stream, output_width)

### Function: __init__(self, depth)

### Function: __init__(self)

### Function: enq(self, group)

### Function: deq(self)

### Function: remove(self, group)

### Function: __init__(self, value)

### Function: _repr_pretty_(self, p, cycle)

### Function: __init__(__self, __name)

### Function: factory(cls, name)

### Function: _repr_pretty_(self, p, cycle)

### Function: __init__(self, value)

### Function: _repr_pretty_(self, p, cycle)

### Function: inner(obj, p, cycle)

### Function: inner(obj, p, cycle)

### Function: inner(obj, p, cycle)

### Function: __init__(self, value)

### Function: _repr_pretty_(self, p, cycle)

## Class: Foo

### Function: inner()

### Function: new_item()

### Function: __init__(self)

### Function: get_foo(self)
