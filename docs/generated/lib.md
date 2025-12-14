## AI Summary

A file named lib.py.


## Class: QuoteStyle

**Description:** Controls how strings will be quoted during encoding.

By default, for compatibility with the `json` module and older versions of
`json5`, strings (not being used as keys and that are legal identifiers)
will always be double-quoted, and any double quotes in the string will be
escaped. This is `QuoteStyle.ALWAYS_DOUBLE`.  If you pass
`QuoteStyle.ALWAYS_SINGLE`, then strings will always be single-quoted, and
any single quotes in the string will be escaped.  If you pass
`QuoteStyle.PREFER_DOUBLE`, then the behavior is the same as ALWAYS_DOUBLE
and strings will be double-quoted *unless* the string contains more double
quotes than single quotes, in which case the string will be single-quoted
and single quotes will be escaped. If you pass `QuoteStyle.PREFER_SINGLE`,
then the behavior is the same as ALWAYS_SINGLE and strings will be
single-quoted *unless* the string contains more single quotes than double
quotes, in which case the string will be double-quoted and any double
quotes will be escaped.

*Note:* PREFER_DOUBLE and PREFER_SINGLE can impact performance, since in
order to know which encoding to use you have to iterate over the entire
string to count the number of single and double quotes. The codes guesses
at an encoding while doing so, but if it guess wrong, the entire string has
to be re-encoded, which will slow things down. If you are very concerned
about performance (a) you probably shouldn't be using this library in the
first place, because it just isn't very fast, and (b) you should use
ALWAYS_DOUBLE or ALWAYS_SINGLE, which won't have this issue.

### Function: load(fp)

**Description:** Deserialize ``fp`` (a ``.read()``-supporting file-like object
containing a JSON document) to a Python object.

Supports almost the same arguments as ``json.load()`` except that:
    - the `cls` keyword is ignored.
    - an extra `allow_duplicate_keys` parameter supports checking for
      duplicate keys in a object; by default, this is True for
      compatibility with ``json.load()``, but if set to False and
      the object contains duplicate keys, a ValueError will be raised.
    - an extra `consume_trailing` parameter specifies whether to
      consume any trailing characters after a valid object has been
      parsed. By default, this value is True and the only legal
      trailing characters are whitespace. If this value is set to False,
      parsing will stop when a valid object has been parsed and any
      trailing characters in the string will be ignored.
    - an extra `start` parameter specifies the zero-based offset into the
      file to start parsing at. If `start` is None, parsing will
      start at the current position in the file, and line number
      and column values will be reported as if starting from the
      beginning of the file; If `start` is not None,
      `load` will seek to zero and then read (and discard) the
      appropriate number of characters before beginning parsing;
      the file must be seekable for this to work correctly.

You can use `load(..., consume_trailing=False)` to repeatedly read
values from a file. However, in the current implementation `load` does
this by reading the entire file into memory before doing anything, so
it is not very efficient.

Raises
    - `ValueError` if given an invalid document. This is different
      from the `json` module, which raises `json.JSONDecodeError`.
    - `UnicodeDecodeError` if given a byte string that is not a
      legal UTF-8 document (or the equivalent, if using a different
      `encoding`). This matches the `json` module.

### Function: loads(s)

**Description:** Deserialize ``s`` (a string containing a JSON5 document) to a Python
object.

Supports the same arguments as ``json.load()`` except that:
    - the `cls` keyword is ignored.
    - an extra `allow_duplicate_keys` parameter supports checking for
      duplicate keys in a object; by default, this is True for
      compatibility with ``json.load()``, but if set to False and
      the object contains duplicate keys, a ValueError will be raised.
    - an extra `consume_trailing` parameter specifies whether to
      consume any trailing characters after a valid object has been
      parsed. By default, this value is True and the only legal
      trailing characters are whitespace. If this value is set to False,
      parsing will stop when a valid object has been parsed and any
      trailing characters in the string will be ignored.
    - an extra `start` parameter specifies the zero-based offset into the
      string to start parsing at.

Raises
    - `ValueError` if given an invalid document. This is different
      from the `json` module, which raises `json.JSONDecodeError`.
    - `UnicodeDecodeError` if given a byte string that is not a
      legal UTF-8 document (or the equivalent, if using a different
      `encoding`). This matches the `json` module.

### Function: parse(s)

**Description:** Parse ```s``, returning positional information along with a value.

This works exactly like `loads()`, except that (a) it returns the
position in the string where the parsing stopped (either due to
hitting an error or parsing a valid value) and any error as a string,
(b) it takes an optional `consume_trailing` parameter that says whether
to keep parsing the string after a valid value has been parsed; if True
(the default), any trailing characters must be whitespace. If False,
parsing stops when a valid value has been reached, (c) it takes an
optional `start` parameter that specifies a zero-based offset to start
parsing from in the string, and (d) the return value is different, as
described below.

`parse()` is useful if you have a string that might contain multiple
values and you need to extract all of them; you can do so by repeatedly
calling `parse`, setting `start` to the value returned in `position`
from the previous call.

Returns a tuple of (value, error_string, position). If the string
    was a legal value, `value` will be the deserialized value,
    `error_string` will be `None`, and `position` will be one
    past the zero-based offset where the parser stopped reading.
    If the string was not a legal value,
    `value` will be `None`, `error_string` will be the string value
    of the exception that would've been raised, and `position` will
    be the zero-based farthest offset into the string where the parser
    hit an error.

Raises:
    - `UnicodeDecodeError` if given a byte string that is not a
      legal UTF-8 document (or the equivalent, if using a different
      `encoding`). This matches the `json` module.

Note that this does *not* raise a `ValueError`; instead any error is
returned as the second value in the tuple.

You can use this method to read in a series of values from a string
`s` as follows:

>>> import json5
>>> s = '1 2 3 4'
>>> values = []
>>> start = 0
>>> while True:
...     v, err, pos = json5.parse(s, start=start, consume_trailing=False)
...     if v:
...         values.append(v)
...         start = pos
...         if start == len(s) or s[start:].isspace():
...             # Reached the end of the string (ignoring trailing
...             # whitespace
...             break
...         continue
...     raise ValueError(err)
>>> values
[1, 2, 3, 4]

### Function: _convert(ast, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, allow_duplicate_keys)

### Function: _walk_ast(el, dictify, parse_float, parse_int, parse_constant)

### Function: dump(obj, fp)

**Description:** Serialize ``obj`` to a JSON5-formatted stream to ``fp``,
a ``.write()``-supporting file-like object.

Supports the same arguments as ``dumps()``, below.

Calling ``dump(obj, fp, quote_keys=True, trailing_commas=False,                    allow_duplicate_keys=True)``
should produce exactly the same output as ``json.dump(obj, fp).``

### Function: dumps(obj)

**Description:** Serialize ``obj`` to a JSON5-formatted string.

Supports the same arguments as ``json.dumps()``, except that:

- The ``encoding`` keyword is ignored; Unicode strings are always written.
- By default, object keys that are legal identifiers are not quoted; if you
  pass ``quote_keys=True``, they will be.
- By default, if lists and objects span multiple lines of output (i.e.,
  when ``indent`` >=0), the last item will have a trailing comma after it.
  If you pass ``trailing_commas=False``, it will not.
- If you use a number, a boolean, or ``None`` as a key value in a dict, it
  will be converted to the corresponding JSON string value, e.g.  "1",
  "true", or "null". By default, ``dump()`` will match the `json` modules
  behavior and produce malformed JSON if you mix keys of different types
  that have the same converted value; e.g., ``{1: "foo", "1": "bar"}``
  produces '{"1": "foo", "1": "bar"}', an object with duplicated keys. If
  you pass ``allow_duplicate_keys=False``, an exception will be raised
  instead.
- If `quote_keys` is true, then keys of objects will be enclosed in quotes,
  as in regular JSON. Otheriwse, keys will not be enclosed in quotes unless
  they contain whitespace.
- If `trailing_commas` is false, then commas will not be inserted after the
  final elements of objects and arrays, as in regular JSON.  Otherwise,
  such commas will be inserted.
- If `allow_duplicate_keys` is false, then only the last entry with a given
  key will be written. Otherwise, all entries with the same key will be
  written.
- `quote_style` controls how strings are encoded. See the documentation
  for the `QuoteStyle` class, above, for how this is used.

  *Note*: Strings that are being used as unquoted keys are not affected
  by this parameter and remain unquoted.

  *`quote_style` was added in version 0.10.0*.

Other keyword arguments are allowed and will be passed to the
encoder so custom encoders can get them, but otherwise they will
be ignored in an attempt to provide some amount of forward-compatibility.

*Note:* the standard JSON module explicitly calls `int.__repr(obj)__`
and `float.__repr(obj)__` to encode ints and floats, thereby bypassing
any custom representations you might have for objects that are subclasses
of ints and floats, and, for compatibility, JSON5 does the same thing.
To override this behavior, create a subclass of JSON5Encoder
that overrides `encode()` and handles your custom representation.

For example:

```
>>> import json5
>>> from typing import Any, Set
>>>
>>> class Hex(int):
...    def __repr__(self):
...        return hex(self)
>>>
>>> class CustomEncoder(json5.JSON5Encoder):
...    def encode(
...        self, obj: Any, seen: Set, level: int, *, as_key: bool
...    ) -> str:
...        if isinstance(obj, Hex):
...            return repr(obj)
...        return super().encode(obj, seen, level, as_key=as_key)
...
>>> json5.dumps([20, Hex(20)], cls=CustomEncoder)
'[20, 0x14]'

```

*Note:* calling ``dumps(obj, quote_keys=True, trailing_commas=False,                             allow_duplicate_keys=True)``
should produce exactly the same output as ``json.dumps(obj).``

## Class: JSON5Encoder

### Function: _raise_type_error(obj)

### Function: _fp_constant_parser(s)

### Function: _dictify(pairs)

### Function: __init__(self)

**Description:** Provides a class that may be overridden to customize the behavior
of `dumps()`. The keyword args are the same as for that function.
*Added in version 0.10.0

### Function: default(self, obj)

**Description:** Provides a last-ditch option to encode a value that the encoder
doesn't otherwise recognize, by converting `obj` to a value that
*can* (and will) be serialized by the other methods in the class.

Note: this must not return a serialized value (i.e., string)
directly, as that'll result in a doubly-encoded value.

### Function: encode(self, obj, seen, level)

**Description:** Returns an JSON5-encoded version of an arbitrary object. This can
be used to provide customized serialization of objects. Overridden
methods of this class should handle their custom objects and then
fall back to super.encode() if they've been passed a normal object.

`seen` is used for duplicate object tracking when `check_circular`
is True.

`level` represents the current indentation level, which increases
by one for each recursive invocation of encode (i.e., whenever
we're encoding the values of a dict or a list).

May raise `TypeError` if the object is the wrong type to be
encoded (i.e., your custom routine can't handle it either), and
`ValueError` if there's something wrong with the value, e.g.
a float value of NaN when `allow_nan` is false.

If `as_key` is true, the return value should be a double-quoted string
representation of the object, unless obj is a string that can be an
identifier (and quote_keys is false and obj isn't a reserved word).
If the object should not be used as a key, `TypeError` should be
raised; that allows the base implementation to implement `skipkeys`
properly.

### Function: _encode_basic_type(self, obj)

**Description:** Returns None if the object is not a basic type.

### Function: _encode_int(self, obj)

### Function: _encode_float(self, obj)

### Function: _encode_str(self, obj)

### Function: _encode_quoted_str(self, obj, quote_style)

**Description:** Returns a quoted string with a minimal number of escaped quotes.

### Function: _escape_ch(self, ch)

**Description:** Returns the backslash-escaped representation of the char.

### Function: _encode_non_basic_type(self, obj, seen, level)

### Function: _encode_dict(self, obj, seen, level)

### Function: _encode_array(self, obj, seen, level)

### Function: _spacers(self, level)

### Function: is_identifier(self, key)

**Description:** Returns whether the string could be used as a legal
EcmaScript/JavaScript identifier.

There should normally be no reason to override this, unless
the definition of identifiers change in later versions of the
JSON5 spec and this implementation hasn't been updated to handle
the changes yet.

### Function: _is_id_start(self, ch)

### Function: _is_id_continue(self, ch)

### Function: is_reserved_word(self, key)

**Description:** Returns whether the key is a reserved word.

There should normally be no need to override this, unless there
have been reserved words added in later versions of the JSON5
spec and this implementation has not yet been updated to handle
the changes yet.
