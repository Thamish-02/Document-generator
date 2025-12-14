## AI Summary

A file named arrayprint.py.


### Function: _make_options_dict(precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr, sign, formatter, floatmode, legacy, override_repr)

**Description:** Make a dictionary out of the non-None arguments, plus conversion of
*legacy* and sanity checks.

### Function: set_printoptions(precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr, formatter, sign, floatmode)

**Description:** Set printing options.

These options determine the way floating point numbers, arrays and
other NumPy objects are displayed.

Parameters
----------
precision : int or None, optional
    Number of digits of precision for floating point output (default 8).
    May be None if `floatmode` is not `fixed`, to print as many digits as
    necessary to uniquely specify the value.
threshold : int, optional
    Total number of array elements which trigger summarization
    rather than full repr (default 1000).
    To always use the full repr without summarization, pass `sys.maxsize`.
edgeitems : int, optional
    Number of array items in summary at beginning and end of
    each dimension (default 3).
linewidth : int, optional
    The number of characters per line for the purpose of inserting
    line breaks (default 75).
suppress : bool, optional
    If True, always print floating point numbers using fixed point
    notation, in which case numbers equal to zero in the current precision
    will print as zero.  If False, then scientific notation is used when
    absolute value of the smallest number is < 1e-4 or the ratio of the
    maximum absolute value to the minimum is > 1e3. The default is False.
nanstr : str, optional
    String representation of floating point not-a-number (default nan).
infstr : str, optional
    String representation of floating point infinity (default inf).
sign : string, either '-', '+', or ' ', optional
    Controls printing of the sign of floating-point types. If '+', always
    print the sign of positive values. If ' ', always prints a space
    (whitespace character) in the sign position of positive values.  If
    '-', omit the sign character of positive values. (default '-')

    .. versionchanged:: 2.0
         The sign parameter can now be an integer type, previously
         types were floating-point types.

formatter : dict of callables, optional
    If not None, the keys should indicate the type(s) that the respective
    formatting function applies to.  Callables should return a string.
    Types that are not specified (by their corresponding keys) are handled
    by the default formatters.  Individual types for which a formatter
    can be set are:

    - 'bool'
    - 'int'
    - 'timedelta' : a `numpy.timedelta64`
    - 'datetime' : a `numpy.datetime64`
    - 'float'
    - 'longfloat' : 128-bit floats
    - 'complexfloat'
    - 'longcomplexfloat' : composed of two 128-bit floats
    - 'numpystr' : types `numpy.bytes_` and `numpy.str_`
    - 'object' : `np.object_` arrays

    Other keys that can be used to set a group of types at once are:

    - 'all' : sets all types
    - 'int_kind' : sets 'int'
    - 'float_kind' : sets 'float' and 'longfloat'
    - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
    - 'str_kind' : sets 'numpystr'
floatmode : str, optional
    Controls the interpretation of the `precision` option for
    floating-point types. Can take the following values
    (default maxprec_equal):

    * 'fixed': Always print exactly `precision` fractional digits,
            even if this would print more or fewer digits than
            necessary to specify the value uniquely.
    * 'unique': Print the minimum number of fractional digits necessary
            to represent each value uniquely. Different elements may
            have a different number of digits. The value of the
            `precision` option is ignored.
    * 'maxprec': Print at most `precision` fractional digits, but if
            an element can be uniquely represented with fewer digits
            only print it with that many.
    * 'maxprec_equal': Print at most `precision` fractional digits,
            but if every element in the array can be uniquely
            represented with an equal number of fewer digits, use that
            many digits for all elements.
legacy : string or `False`, optional
    If set to the string ``'1.13'`` enables 1.13 legacy printing mode. This
    approximates numpy 1.13 print output by including a space in the sign
    position of floats and different behavior for 0d arrays. This also
    enables 1.21 legacy printing mode (described below).

    If set to the string ``'1.21'`` enables 1.21 legacy printing mode. This
    approximates numpy 1.21 print output of complex structured dtypes
    by not inserting spaces after commas that separate fields and after
    colons.

    If set to ``'1.25'`` approximates printing of 1.25 which mainly means
    that numeric scalars are printed without their type information, e.g.
    as ``3.0`` rather than ``np.float64(3.0)``.

    If set to ``'2.1'``, shape information is not given when arrays are
    summarized (i.e., multiple elements replaced with ``...``).

    If set to `False`, disables legacy mode.

    Unrecognized strings will be ignored with a warning for forward
    compatibility.

    .. versionchanged:: 1.22.0
    .. versionchanged:: 2.2

override_repr: callable, optional
    If set a passed function will be used for generating arrays' repr.
    Other options will be ignored.

See Also
--------
get_printoptions, printoptions, array2string

Notes
-----
`formatter` is always reset with a call to `set_printoptions`.

Use `printoptions` as a context manager to set the values temporarily.

Examples
--------
Floating point precision can be set:

>>> import numpy as np
>>> np.set_printoptions(precision=4)
>>> np.array([1.123456789])
[1.1235]

Long arrays can be summarised:

>>> np.set_printoptions(threshold=5)
>>> np.arange(10)
array([0, 1, 2, ..., 7, 8, 9], shape=(10,))

Small results can be suppressed:

>>> eps = np.finfo(float).eps
>>> x = np.arange(4.)
>>> x**2 - (x + eps)**2
array([-4.9304e-32, -4.4409e-16,  0.0000e+00,  0.0000e+00])
>>> np.set_printoptions(suppress=True)
>>> x**2 - (x + eps)**2
array([-0., -0.,  0.,  0.])

A custom formatter can be used to display array elements as desired:

>>> np.set_printoptions(formatter={'all':lambda x: 'int: '+str(-x)})
>>> x = np.arange(3)
>>> x
array([int: 0, int: -1, int: -2])
>>> np.set_printoptions()  # formatter gets reset
>>> x
array([0, 1, 2])

To put back the default options, you can use:

>>> np.set_printoptions(edgeitems=3, infstr='inf',
... linewidth=75, nanstr='nan', precision=8,
... suppress=False, threshold=1000, formatter=None)

Also to temporarily override options, use `printoptions`
as a context manager:

>>> with np.printoptions(precision=2, suppress=True, threshold=5):
...     np.linspace(0, 10, 10)
array([ 0.  ,  1.11,  2.22, ...,  7.78,  8.89, 10.  ], shape=(10,))

### Function: _set_printoptions(precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr, formatter, sign, floatmode)

### Function: get_printoptions()

**Description:** Return the current print options.

Returns
-------
print_opts : dict
    Dictionary of current print options with keys

    - precision : int
    - threshold : int
    - edgeitems : int
    - linewidth : int
    - suppress : bool
    - nanstr : str
    - infstr : str
    - sign : str
    - formatter : dict of callables
    - floatmode : str
    - legacy : str or False

    For a full description of these options, see `set_printoptions`.

See Also
--------
set_printoptions, printoptions

Examples
--------
>>> import numpy as np

>>> np.get_printoptions()
{'edgeitems': 3, 'threshold': 1000, ..., 'override_repr': None}

>>> np.get_printoptions()['linewidth']
75
>>> np.set_printoptions(linewidth=100)
>>> np.get_printoptions()['linewidth']
100

### Function: _get_legacy_print_mode()

**Description:** Return the legacy print mode as an int.

### Function: printoptions()

**Description:** Context manager for setting print options.

Set print options for the scope of the `with` block, and restore the old
options at the end. See `set_printoptions` for the full description of
available options.

Examples
--------
>>> import numpy as np

>>> from numpy.testing import assert_equal
>>> with np.printoptions(precision=2):
...     np.array([2.0]) / 3
array([0.67])

The `as`-clause of the `with`-statement gives the current print options:

>>> with np.printoptions(precision=2) as opts:
...      assert_equal(opts, np.get_printoptions())

See Also
--------
set_printoptions, get_printoptions

### Function: _leading_trailing(a, edgeitems, index)

**Description:** Keep only the N-D corners (leading and trailing edges) of an array.

Should be passed a base-class ndarray, since it makes no guarantees about
preserving subclasses.

### Function: _object_format(o)

**Description:** Object arrays containing lists should be printed unambiguously 

### Function: repr_format(x)

### Function: str_format(x)

### Function: _get_formatdict(data)

### Function: _get_format_function(data)

**Description:** find the right formatting function for the dtype_

### Function: _recursive_guard(fillvalue)

**Description:** Like the python 3.2 reprlib.recursive_repr, but forwards *args and **kwargs

Decorates a function such that if it calls itself with the same first
argument, it returns `fillvalue` instead of recursing.

Largely copied from reprlib.recursive_repr

### Function: _array2string(a, options, separator, prefix)

### Function: _array2string_dispatcher(a, max_line_width, precision, suppress_small, separator, prefix, style, formatter, threshold, edgeitems, sign, floatmode, suffix)

### Function: array2string(a, max_line_width, precision, suppress_small, separator, prefix, style, formatter, threshold, edgeitems, sign, floatmode, suffix)

**Description:** Return a string representation of an array.

Parameters
----------
a : ndarray
    Input array.
max_line_width : int, optional
    Inserts newlines if text is longer than `max_line_width`.
    Defaults to ``numpy.get_printoptions()['linewidth']``.
precision : int or None, optional
    Floating point precision.
    Defaults to ``numpy.get_printoptions()['precision']``.
suppress_small : bool, optional
    Represent numbers "very close" to zero as zero; default is False.
    Very close is defined by precision: if the precision is 8, e.g.,
    numbers smaller (in absolute value) than 5e-9 are represented as
    zero.
    Defaults to ``numpy.get_printoptions()['suppress']``.
separator : str, optional
    Inserted between elements.
prefix : str, optional
suffix : str, optional
    The length of the prefix and suffix strings are used to respectively
    align and wrap the output. An array is typically printed as::

      prefix + array2string(a) + suffix

    The output is left-padded by the length of the prefix string, and
    wrapping is forced at the column ``max_line_width - len(suffix)``.
    It should be noted that the content of prefix and suffix strings are
    not included in the output.
style : _NoValue, optional
    Has no effect, do not use.

    .. deprecated:: 1.14.0
formatter : dict of callables, optional
    If not None, the keys should indicate the type(s) that the respective
    formatting function applies to.  Callables should return a string.
    Types that are not specified (by their corresponding keys) are handled
    by the default formatters.  Individual types for which a formatter
    can be set are:

    - 'bool'
    - 'int'
    - 'timedelta' : a `numpy.timedelta64`
    - 'datetime' : a `numpy.datetime64`
    - 'float'
    - 'longfloat' : 128-bit floats
    - 'complexfloat'
    - 'longcomplexfloat' : composed of two 128-bit floats
    - 'void' : type `numpy.void`
    - 'numpystr' : types `numpy.bytes_` and `numpy.str_`

    Other keys that can be used to set a group of types at once are:

    - 'all' : sets all types
    - 'int_kind' : sets 'int'
    - 'float_kind' : sets 'float' and 'longfloat'
    - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
    - 'str_kind' : sets 'numpystr'
threshold : int, optional
    Total number of array elements which trigger summarization
    rather than full repr.
    Defaults to ``numpy.get_printoptions()['threshold']``.
edgeitems : int, optional
    Number of array items in summary at beginning and end of
    each dimension.
    Defaults to ``numpy.get_printoptions()['edgeitems']``.
sign : string, either '-', '+', or ' ', optional
    Controls printing of the sign of floating-point types. If '+', always
    print the sign of positive values. If ' ', always prints a space
    (whitespace character) in the sign position of positive values.  If
    '-', omit the sign character of positive values.
    Defaults to ``numpy.get_printoptions()['sign']``.

    .. versionchanged:: 2.0
         The sign parameter can now be an integer type, previously
         types were floating-point types.

floatmode : str, optional
    Controls the interpretation of the `precision` option for
    floating-point types.
    Defaults to ``numpy.get_printoptions()['floatmode']``.
    Can take the following values:

    - 'fixed': Always print exactly `precision` fractional digits,
      even if this would print more or fewer digits than
      necessary to specify the value uniquely.
    - 'unique': Print the minimum number of fractional digits necessary
      to represent each value uniquely. Different elements may
      have a different number of digits.  The value of the
      `precision` option is ignored.
    - 'maxprec': Print at most `precision` fractional digits, but if
      an element can be uniquely represented with fewer digits
      only print it with that many.
    - 'maxprec_equal': Print at most `precision` fractional digits,
      but if every element in the array can be uniquely
      represented with an equal number of fewer digits, use that
      many digits for all elements.
legacy : string or `False`, optional
    If set to the string ``'1.13'`` enables 1.13 legacy printing mode. This
    approximates numpy 1.13 print output by including a space in the sign
    position of floats and different behavior for 0d arrays. If set to
    `False`, disables legacy mode. Unrecognized strings will be ignored
    with a warning for forward compatibility.

Returns
-------
array_str : str
    String representation of the array.

Raises
------
TypeError
    if a callable in `formatter` does not return a string.

See Also
--------
array_str, array_repr, set_printoptions, get_printoptions

Notes
-----
If a formatter is specified for a certain type, the `precision` keyword is
ignored for that type.

This is a very flexible function; `array_repr` and `array_str` are using
`array2string` internally so keywords with the same name should work
identically in all three functions.

Examples
--------
>>> import numpy as np
>>> x = np.array([1e-16,1,2,3])
>>> np.array2string(x, precision=2, separator=',',
...                       suppress_small=True)
'[0.,1.,2.,3.]'

>>> x  = np.arange(3.)
>>> np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
'[0.00 1.00 2.00]'

>>> x  = np.arange(3)
>>> np.array2string(x, formatter={'int':lambda x: hex(x)})
'[0x0 0x1 0x2]'

### Function: _extendLine(s, line, word, line_width, next_line_prefix, legacy)

### Function: _extendLine_pretty(s, line, word, line_width, next_line_prefix, legacy)

**Description:** Extends line with nicely formatted (possibly multi-line) string ``word``.

### Function: _formatArray(a, format_function, line_width, next_line_prefix, separator, edge_items, summary_insert, legacy)

**Description:** formatArray is designed for two modes of operation:

1. Full output

2. Summarized output

### Function: _none_or_positive_arg(x, name)

## Class: FloatingFormat

**Description:** Formatter for subtypes of np.floating 

### Function: format_float_scientific(x, precision, unique, trim, sign, pad_left, exp_digits, min_digits)

**Description:** Format a floating-point scalar as a decimal string in scientific notation.

Provides control over rounding, trimming and padding. Uses and assumes
IEEE unbiased rounding. Uses the "Dragon4" algorithm.

Parameters
----------
x : python float or numpy floating scalar
    Value to format.
precision : non-negative integer or None, optional
    Maximum number of digits to print. May be None if `unique` is
    `True`, but must be an integer if unique is `False`.
unique : boolean, optional
    If `True`, use a digit-generation strategy which gives the shortest
    representation which uniquely identifies the floating-point number from
    other values of the same type, by judicious rounding. If `precision`
    is given fewer digits than necessary can be printed. If `min_digits`
    is given more can be printed, in which cases the last digit is rounded
    with unbiased rounding.
    If `False`, digits are generated as if printing an infinite-precision
    value and stopping after `precision` digits, rounding the remaining
    value with unbiased rounding
trim : one of 'k', '.', '0', '-', optional
    Controls post-processing trimming of trailing digits, as follows:

    * 'k' : keep trailing zeros, keep decimal point (no trimming)
    * '.' : trim all trailing zeros, leave decimal point
    * '0' : trim all but the zero before the decimal point. Insert the
      zero if it is missing.
    * '-' : trim trailing zeros and any trailing decimal point
sign : boolean, optional
    Whether to show the sign for positive values.
pad_left : non-negative integer, optional
    Pad the left side of the string with whitespace until at least that
    many characters are to the left of the decimal point.
exp_digits : non-negative integer, optional
    Pad the exponent with zeros until it contains at least this
    many digits. If omitted, the exponent will be at least 2 digits.
min_digits : non-negative integer or None, optional
    Minimum number of digits to print. This only has an effect for
    `unique=True`. In that case more digits than necessary to uniquely
    identify the value may be printed and rounded unbiased.

    .. versionadded:: 1.21.0

Returns
-------
rep : string
    The string representation of the floating point value

See Also
--------
format_float_positional

Examples
--------
>>> import numpy as np
>>> np.format_float_scientific(np.float32(np.pi))
'3.1415927e+00'
>>> s = np.float32(1.23e24)
>>> np.format_float_scientific(s, unique=False, precision=15)
'1.230000071797338e+24'
>>> np.format_float_scientific(s, exp_digits=4)
'1.23e+0024'

### Function: format_float_positional(x, precision, unique, fractional, trim, sign, pad_left, pad_right, min_digits)

**Description:** Format a floating-point scalar as a decimal string in positional notation.

Provides control over rounding, trimming and padding. Uses and assumes
IEEE unbiased rounding. Uses the "Dragon4" algorithm.

Parameters
----------
x : python float or numpy floating scalar
    Value to format.
precision : non-negative integer or None, optional
    Maximum number of digits to print. May be None if `unique` is
    `True`, but must be an integer if unique is `False`.
unique : boolean, optional
    If `True`, use a digit-generation strategy which gives the shortest
    representation which uniquely identifies the floating-point number from
    other values of the same type, by judicious rounding. If `precision`
    is given fewer digits than necessary can be printed, or if `min_digits`
    is given more can be printed, in which cases the last digit is rounded
    with unbiased rounding.
    If `False`, digits are generated as if printing an infinite-precision
    value and stopping after `precision` digits, rounding the remaining
    value with unbiased rounding
fractional : boolean, optional
    If `True`, the cutoffs of `precision` and `min_digits` refer to the
    total number of digits after the decimal point, including leading
    zeros.
    If `False`, `precision` and `min_digits` refer to the total number of
    significant digits, before or after the decimal point, ignoring leading
    zeros.
trim : one of 'k', '.', '0', '-', optional
    Controls post-processing trimming of trailing digits, as follows:

    * 'k' : keep trailing zeros, keep decimal point (no trimming)
    * '.' : trim all trailing zeros, leave decimal point
    * '0' : trim all but the zero before the decimal point. Insert the
      zero if it is missing.
    * '-' : trim trailing zeros and any trailing decimal point
sign : boolean, optional
    Whether to show the sign for positive values.
pad_left : non-negative integer, optional
    Pad the left side of the string with whitespace until at least that
    many characters are to the left of the decimal point.
pad_right : non-negative integer, optional
    Pad the right side of the string with whitespace until at least that
    many characters are to the right of the decimal point.
min_digits : non-negative integer or None, optional
    Minimum number of digits to print. Only has an effect if `unique=True`
    in which case additional digits past those necessary to uniquely
    identify the value may be printed, rounding the last additional digit.

    .. versionadded:: 1.21.0

Returns
-------
rep : string
    The string representation of the floating point value

See Also
--------
format_float_scientific

Examples
--------
>>> import numpy as np
>>> np.format_float_positional(np.float32(np.pi))
'3.1415927'
>>> np.format_float_positional(np.float16(np.pi))
'3.14'
>>> np.format_float_positional(np.float16(0.3))
'0.3'
>>> np.format_float_positional(np.float16(0.3), unique=False, precision=10)
'0.3000488281'

## Class: IntegerFormat

## Class: BoolFormat

## Class: ComplexFloatingFormat

**Description:** Formatter for subtypes of np.complexfloating 

## Class: _TimelikeFormat

## Class: DatetimeFormat

## Class: TimedeltaFormat

## Class: SubArrayFormat

## Class: StructuredVoidFormat

**Description:** Formatter for structured np.void objects.

This does not work on structured alias types like
np.dtype(('i4', 'i2,i2')), as alias scalars lose their field information,
and the implementation relies upon np.void.__getitem__.

### Function: _void_scalar_to_string(x, is_repr)

**Description:** Implements the repr for structured-void scalars. It is called from the
scalartypes.c.src code, and is placed here because it uses the elementwise
formatters defined above.

### Function: dtype_is_implied(dtype)

**Description:** Determine if the given dtype is implied by the representation
of its values.

Parameters
----------
dtype : dtype
    Data type

Returns
-------
implied : bool
    True if the dtype is implied by the representation of its values.

Examples
--------
>>> import numpy as np
>>> np._core.arrayprint.dtype_is_implied(int)
True
>>> np.array([1, 2, 3], int)
array([1, 2, 3])
>>> np._core.arrayprint.dtype_is_implied(np.int8)
False
>>> np.array([1, 2, 3], np.int8)
array([1, 2, 3], dtype=int8)

### Function: dtype_short_repr(dtype)

**Description:** Convert a dtype to a short form which evaluates to the same dtype.

The intent is roughly that the following holds

>>> from numpy import *
>>> dt = np.int64([1, 2]).dtype
>>> assert eval(dtype_short_repr(dt)) == dt

### Function: _array_repr_implementation(arr, max_line_width, precision, suppress_small, array2string)

**Description:** Internal version of array_repr() that allows overriding array2string.

### Function: _array_repr_dispatcher(arr, max_line_width, precision, suppress_small)

### Function: array_repr(arr, max_line_width, precision, suppress_small)

**Description:** Return the string representation of an array.

Parameters
----------
arr : ndarray
    Input array.
max_line_width : int, optional
    Inserts newlines if text is longer than `max_line_width`.
    Defaults to ``numpy.get_printoptions()['linewidth']``.
precision : int, optional
    Floating point precision.
    Defaults to ``numpy.get_printoptions()['precision']``.
suppress_small : bool, optional
    Represent numbers "very close" to zero as zero; default is False.
    Very close is defined by precision: if the precision is 8, e.g.,
    numbers smaller (in absolute value) than 5e-9 are represented as
    zero.
    Defaults to ``numpy.get_printoptions()['suppress']``.

Returns
-------
string : str
  The string representation of an array.

See Also
--------
array_str, array2string, set_printoptions

Examples
--------
>>> import numpy as np
>>> np.array_repr(np.array([1,2]))
'array([1, 2])'
>>> np.array_repr(np.ma.array([0.]))
'MaskedArray([0.])'
>>> np.array_repr(np.array([], np.int32))
'array([], dtype=int32)'

>>> x = np.array([1e-6, 4e-7, 2, 3])
>>> np.array_repr(x, precision=6, suppress_small=True)
'array([0.000001,  0.      ,  2.      ,  3.      ])'

### Function: _guarded_repr_or_str(v)

### Function: _array_str_implementation(a, max_line_width, precision, suppress_small, array2string)

**Description:** Internal version of array_str() that allows overriding array2string.

### Function: _array_str_dispatcher(a, max_line_width, precision, suppress_small)

### Function: array_str(a, max_line_width, precision, suppress_small)

**Description:** Return a string representation of the data in an array.

The data in the array is returned as a single string.  This function is
similar to `array_repr`, the difference being that `array_repr` also
returns information on the kind of array and its data type.

Parameters
----------
a : ndarray
    Input array.
max_line_width : int, optional
    Inserts newlines if text is longer than `max_line_width`.
    Defaults to ``numpy.get_printoptions()['linewidth']``.
precision : int, optional
    Floating point precision.
    Defaults to ``numpy.get_printoptions()['precision']``.
suppress_small : bool, optional
    Represent numbers "very close" to zero as zero; default is False.
    Very close is defined by precision: if the precision is 8, e.g.,
    numbers smaller (in absolute value) than 5e-9 are represented as
    zero.
    Defaults to ``numpy.get_printoptions()['suppress']``.

See Also
--------
array2string, array_repr, set_printoptions

Examples
--------
>>> import numpy as np
>>> np.array_str(np.arange(3))
'[0 1 2]'

### Function: indirect(x)

### Function: decorating_function(f)

### Function: recurser(index, hanging_indent, curr_width)

**Description:** By using this local function, we don't need to recurse with all the
arguments. Since this function is not created recursively, the cost is
not significant

### Function: __init__(self, data, precision, floatmode, suppress_small, sign)

### Function: fillFormat(self, data)

### Function: __call__(self, x)

### Function: __init__(self, data, sign)

### Function: __call__(self, x)

### Function: __init__(self, data)

### Function: __call__(self, x)

### Function: __init__(self, x, precision, floatmode, suppress_small, sign)

### Function: __call__(self, x)

### Function: __init__(self, data)

### Function: _format_non_nat(self, x)

### Function: __call__(self, x)

### Function: __init__(self, x, unit, timezone, casting, legacy)

### Function: __call__(self, x)

### Function: _format_non_nat(self, x)

### Function: _format_non_nat(self, x)

### Function: __init__(self, format_function)

### Function: __call__(self, a)

### Function: format_array(self, a)

### Function: __init__(self, format_functions)

### Function: from_data(cls, data)

**Description:** This is a second way to initialize StructuredVoidFormat,
using the raw data as input. Added to avoid changing
the signature of __init__.

### Function: __call__(self, x)

### Function: wrapper(self)
