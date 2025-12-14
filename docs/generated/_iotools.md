## AI Summary

A file named _iotools.py.


### Function: _decode_line(line, encoding)

**Description:** Decode bytes from binary input streams.

Defaults to decoding from 'latin1'. That differs from the behavior of
np.compat.asunicode that decodes from 'ascii'.

Parameters
----------
line : str or bytes
     Line to be decoded.
encoding : str
     Encoding used to decode `line`.

Returns
-------
decoded_line : str

### Function: _is_string_like(obj)

**Description:** Check whether obj behaves like a string.

### Function: _is_bytes_like(obj)

**Description:** Check whether obj behaves like a bytes object.

### Function: has_nested_fields(ndtype)

**Description:** Returns whether one or several fields of a dtype are nested.

Parameters
----------
ndtype : dtype
    Data-type of a structured array.

Raises
------
AttributeError
    If `ndtype` does not have a `names` attribute.

Examples
--------
>>> import numpy as np
>>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float)])
>>> np.lib._iotools.has_nested_fields(dt)
False

### Function: flatten_dtype(ndtype, flatten_base)

**Description:** Unpack a structured data-type by collapsing nested fields and/or fields
with a shape.

Note that the field names are lost.

Parameters
----------
ndtype : dtype
    The datatype to collapse
flatten_base : bool, optional
   If True, transform a field with a shape into several fields. Default is
   False.

Examples
--------
>>> import numpy as np
>>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
...                ('block', int, (2, 3))])
>>> np.lib._iotools.flatten_dtype(dt)
[dtype('S4'), dtype('float64'), dtype('float64'), dtype('int64')]
>>> np.lib._iotools.flatten_dtype(dt, flatten_base=True)
[dtype('S4'),
 dtype('float64'),
 dtype('float64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64')]

## Class: LineSplitter

**Description:** Object to split a string at a given delimiter or at given places.

Parameters
----------
delimiter : str, int, or sequence of ints, optional
    If a string, character used to delimit consecutive fields.
    If an integer or a sequence of integers, width(s) of each field.
comments : str, optional
    Character used to mark the beginning of a comment. Default is '#'.
autostrip : bool, optional
    Whether to strip each individual field. Default is True.

## Class: NameValidator

**Description:** Object to validate a list of strings to use as field names.

The strings are stripped of any non alphanumeric character, and spaces
are replaced by '_'. During instantiation, the user can define a list
of names to exclude, as well as a list of invalid characters. Names in
the exclusion list are appended a '_' character.

Once an instance has been created, it can be called with a list of
names, and a list of valid names will be created.  The `__call__`
method accepts an optional keyword "default" that sets the default name
in case of ambiguity. By default this is 'f', so that names will
default to `f0`, `f1`, etc.

Parameters
----------
excludelist : sequence, optional
    A list of names to exclude. This list is appended to the default
    list ['return', 'file', 'print']. Excluded names are appended an
    underscore: for example, `file` becomes `file_` if supplied.
deletechars : str, optional
    A string combining invalid characters that must be deleted from the
    names.
case_sensitive : {True, False, 'upper', 'lower'}, optional
    * If True, field names are case-sensitive.
    * If False or 'upper', field names are converted to upper case.
    * If 'lower', field names are converted to lower case.

    The default value is True.
replace_space : '_', optional
    Character(s) used in replacement of white spaces.

Notes
-----
Calling an instance of `NameValidator` is the same as calling its
method `validate`.

Examples
--------
>>> import numpy as np
>>> validator = np.lib._iotools.NameValidator()
>>> validator(['file', 'field2', 'with space', 'CaSe'])
('file_', 'field2', 'with_space', 'CaSe')

>>> validator = np.lib._iotools.NameValidator(excludelist=['excl'],
...                                           deletechars='q',
...                                           case_sensitive=False)
>>> validator(['excl', 'field2', 'no_q', 'with space', 'CaSe'])
('EXCL', 'FIELD2', 'NO_Q', 'WITH_SPACE', 'CASE')

### Function: str2bool(value)

**Description:** Tries to transform a string supposed to represent a boolean to a boolean.

Parameters
----------
value : str
    The string that is transformed to a boolean.

Returns
-------
boolval : bool
    The boolean representation of `value`.

Raises
------
ValueError
    If the string is not 'True' or 'False' (case independent)

Examples
--------
>>> import numpy as np
>>> np.lib._iotools.str2bool('TRUE')
True
>>> np.lib._iotools.str2bool('false')
False

## Class: ConverterError

**Description:** Exception raised when an error occurs in a converter for string values.

## Class: ConverterLockError

**Description:** Exception raised when an attempt is made to upgrade a locked converter.

## Class: ConversionWarning

**Description:** Warning issued when a string converter has a problem.

Notes
-----
In `genfromtxt` a `ConversionWarning` is issued if raising exceptions
is explicitly suppressed with the "invalid_raise" keyword.

## Class: StringConverter

**Description:** Factory class for function transforming a string into another object
(int, float).

After initialization, an instance can be called to transform a string
into another object. If the string is recognized as representing a
missing value, a default value is returned.

Attributes
----------
func : function
    Function used for the conversion.
default : any
    Default value to return when the input corresponds to a missing
    value.
type : type
    Type of the output.
_status : int
    Integer representing the order of the conversion.
_mapper : sequence of tuples
    Sequence of tuples (dtype, function, default value) to evaluate in
    order.
_locked : bool
    Holds `locked` parameter.

Parameters
----------
dtype_or_func : {None, dtype, function}, optional
    If a `dtype`, specifies the input data type, used to define a basic
    function and a default value for missing data. For example, when
    `dtype` is float, the `func` attribute is set to `float` and the
    default value to `np.nan`.  If a function, this function is used to
    convert a string to another object. In this case, it is recommended
    to give an associated default value as input.
default : any, optional
    Value to return by default, that is, when the string to be
    converted is flagged as missing. If not given, `StringConverter`
    tries to supply a reasonable default value.
missing_values : {None, sequence of str}, optional
    ``None`` or sequence of strings indicating a missing value. If ``None``
    then missing values are indicated by empty entries. The default is
    ``None``.
locked : bool, optional
    Whether the StringConverter should be locked to prevent automatic
    upgrade or not. Default is False.

### Function: easy_dtype(ndtype, names, defaultfmt)

**Description:** Convenience function to create a `np.dtype` object.

The function processes the input `dtype` and matches it with the given
names.

Parameters
----------
ndtype : var
    Definition of the dtype. Can be any string or dictionary recognized
    by the `np.dtype` function, or a sequence of types.
names : str or sequence, optional
    Sequence of strings to use as field names for a structured dtype.
    For convenience, `names` can be a string of a comma-separated list
    of names.
defaultfmt : str, optional
    Format string used to define missing names, such as ``"f%i"``
    (default) or ``"fields_%02i"``.
validationargs : optional
    A series of optional arguments used to initialize a
    `NameValidator`.

Examples
--------
>>> import numpy as np
>>> np.lib._iotools.easy_dtype(float)
dtype('float64')
>>> np.lib._iotools.easy_dtype("i4, f8")
dtype([('f0', '<i4'), ('f1', '<f8')])
>>> np.lib._iotools.easy_dtype("i4, f8", defaultfmt="field_%03i")
dtype([('field_000', '<i4'), ('field_001', '<f8')])

>>> np.lib._iotools.easy_dtype((int, float, float), names="a,b,c")
dtype([('a', '<i8'), ('b', '<f8'), ('c', '<f8')])
>>> np.lib._iotools.easy_dtype(float, names="a,b,c")
dtype([('a', '<f8'), ('b', '<f8'), ('c', '<f8')])

### Function: autostrip(self, method)

**Description:** Wrapper to strip each member of the output of `method`.

Parameters
----------
method : function
    Function that takes a single argument and returns a sequence of
    strings.

Returns
-------
wrapped : function
    The result of wrapping `method`. `wrapped` takes a single input
    argument and returns a list of strings that are stripped of
    white-space.

### Function: __init__(self, delimiter, comments, autostrip, encoding)

### Function: _delimited_splitter(self, line)

**Description:** Chop off comments, strip, and split at delimiter. 

### Function: _fixedwidth_splitter(self, line)

### Function: _variablewidth_splitter(self, line)

### Function: __call__(self, line)

### Function: __init__(self, excludelist, deletechars, case_sensitive, replace_space)

### Function: validate(self, names, defaultfmt, nbfields)

**Description:** Validate a list of strings as field names for a structured array.

Parameters
----------
names : sequence of str
    Strings to be validated.
defaultfmt : str, optional
    Default format string, used if validating a given string
    reduces its length to zero.
nbfields : integer, optional
    Final number of validated names, used to expand or shrink the
    initial list of names.

Returns
-------
validatednames : list of str
    The list of validated field names.

Notes
-----
A `NameValidator` instance can be called directly, which is the
same as calling `validate`. For examples, see `NameValidator`.

### Function: __call__(self, names, defaultfmt, nbfields)

### Function: _getdtype(cls, val)

**Description:** Returns the dtype of the input variable.

### Function: _getsubdtype(cls, val)

**Description:** Returns the type of the dtype of the input variable.

### Function: _dtypeortype(cls, dtype)

**Description:** Returns dtype for datetime64 and type of dtype otherwise.

### Function: upgrade_mapper(cls, func, default)

**Description:** Upgrade the mapper of a StringConverter by adding a new function and
its corresponding default.

The input function (or sequence of functions) and its associated
default value (if any) is inserted in penultimate position of the
mapper.  The corresponding type is estimated from the dtype of the
default value.

Parameters
----------
func : var
    Function, or sequence of functions

Examples
--------
>>> import dateutil.parser
>>> import datetime
>>> dateparser = dateutil.parser.parse
>>> defaultdate = datetime.date(2000, 1, 1)
>>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)

### Function: _find_map_entry(cls, dtype)

### Function: __init__(self, dtype_or_func, default, missing_values, locked)

### Function: _loose_call(self, value)

### Function: _strict_call(self, value)

### Function: __call__(self, value)

### Function: _do_upgrade(self)

### Function: upgrade(self, value)

**Description:** Find the best converter for a given string, and return the result.

The supplied string `value` is converted by testing different
converters in order. First the `func` method of the
`StringConverter` instance is tried, if this fails other available
converters are tried.  The order in which these other converters
are tried is determined by the `_status` attribute of the instance.

Parameters
----------
value : str
    The string to convert.

Returns
-------
out : any
    The result of converting `value` with the appropriate converter.

### Function: iterupgrade(self, value)

### Function: update(self, func, default, testing_value, missing_values, locked)

**Description:** Set StringConverter attributes directly.

Parameters
----------
func : function
    Conversion function.
default : any, optional
    Value to return by default, that is, when the string to be
    converted is flagged as missing. If not given,
    `StringConverter` tries to supply a reasonable default value.
testing_value : str, optional
    A string representing a standard input value of the converter.
    This string is used to help defining a reasonable default
    value.
missing_values : {sequence of str, None}, optional
    Sequence of strings indicating a missing value. If ``None``, then
    the existing `missing_values` are cleared. The default is ``''``.
locked : bool, optional
    Whether the StringConverter should be locked to prevent
    automatic upgrade or not. Default is False.

Notes
-----
`update` takes the same parameters as the constructor of
`StringConverter`, except that `func` does not accept a `dtype`
whereas `dtype_or_func` in the constructor does.
