## AI Summary

A file named defchararray.py.


### Function: _binary_op_dispatcher(x1, x2)

### Function: equal(x1, x2)

**Description:** Return (x1 == x2) element-wise.

Unlike `numpy.equal`, this comparison is performed by first
stripping whitespace characters from the end of the string.  This
behavior is provided for backward-compatibility with numarray.

Parameters
----------
x1, x2 : array_like of str or unicode
    Input arrays of the same shape.

Returns
-------
out : ndarray
    Output array of bools.

Examples
--------
>>> import numpy as np
>>> y = "aa "
>>> x = "aa"
>>> np.char.equal(x, y)
array(True)

See Also
--------
not_equal, greater_equal, less_equal, greater, less

### Function: not_equal(x1, x2)

**Description:** Return (x1 != x2) element-wise.

Unlike `numpy.not_equal`, this comparison is performed by first
stripping whitespace characters from the end of the string.  This
behavior is provided for backward-compatibility with numarray.

Parameters
----------
x1, x2 : array_like of str or unicode
    Input arrays of the same shape.

Returns
-------
out : ndarray
    Output array of bools.

See Also
--------
equal, greater_equal, less_equal, greater, less

Examples
--------
>>> import numpy as np
>>> x1 = np.array(['a', 'b', 'c'])
>>> np.char.not_equal(x1, 'b')
array([ True, False,  True])

### Function: greater_equal(x1, x2)

**Description:** Return (x1 >= x2) element-wise.

Unlike `numpy.greater_equal`, this comparison is performed by
first stripping whitespace characters from the end of the string.
This behavior is provided for backward-compatibility with
numarray.

Parameters
----------
x1, x2 : array_like of str or unicode
    Input arrays of the same shape.

Returns
-------
out : ndarray
    Output array of bools.

See Also
--------
equal, not_equal, less_equal, greater, less

Examples
--------
>>> import numpy as np
>>> x1 = np.array(['a', 'b', 'c'])
>>> np.char.greater_equal(x1, 'b')
array([False,  True,  True])

### Function: less_equal(x1, x2)

**Description:** Return (x1 <= x2) element-wise.

Unlike `numpy.less_equal`, this comparison is performed by first
stripping whitespace characters from the end of the string.  This
behavior is provided for backward-compatibility with numarray.

Parameters
----------
x1, x2 : array_like of str or unicode
    Input arrays of the same shape.

Returns
-------
out : ndarray
    Output array of bools.

See Also
--------
equal, not_equal, greater_equal, greater, less

Examples
--------
>>> import numpy as np
>>> x1 = np.array(['a', 'b', 'c'])
>>> np.char.less_equal(x1, 'b')
array([ True,  True, False])

### Function: greater(x1, x2)

**Description:** Return (x1 > x2) element-wise.

Unlike `numpy.greater`, this comparison is performed by first
stripping whitespace characters from the end of the string.  This
behavior is provided for backward-compatibility with numarray.

Parameters
----------
x1, x2 : array_like of str or unicode
    Input arrays of the same shape.

Returns
-------
out : ndarray
    Output array of bools.

See Also
--------
equal, not_equal, greater_equal, less_equal, less

Examples
--------
>>> import numpy as np
>>> x1 = np.array(['a', 'b', 'c'])
>>> np.char.greater(x1, 'b')
array([False, False,  True])

### Function: less(x1, x2)

**Description:** Return (x1 < x2) element-wise.

Unlike `numpy.greater`, this comparison is performed by first
stripping whitespace characters from the end of the string.  This
behavior is provided for backward-compatibility with numarray.

Parameters
----------
x1, x2 : array_like of str or unicode
    Input arrays of the same shape.

Returns
-------
out : ndarray
    Output array of bools.

See Also
--------
equal, not_equal, greater_equal, less_equal, greater

Examples
--------
>>> import numpy as np
>>> x1 = np.array(['a', 'b', 'c'])
>>> np.char.less(x1, 'b')
array([True, False, False])

### Function: multiply(a, i)

**Description:** Return (a * i), that is string multiple concatenation,
element-wise.

Values in ``i`` of less than 0 are treated as 0 (which yields an
empty string).

Parameters
----------
a : array_like, with `np.bytes_` or `np.str_` dtype

i : array_like, with any integer dtype

Returns
-------
out : ndarray
    Output array of str or unicode, depending on input types

Notes
-----
This is a thin wrapper around np.strings.multiply that raises
`ValueError` when ``i`` is not an integer. It only
exists for backwards-compatibility.

Examples
--------
>>> import numpy as np
>>> a = np.array(["a", "b", "c"])
>>> np.strings.multiply(a, 3)
array(['aaa', 'bbb', 'ccc'], dtype='<U3')
>>> i = np.array([1, 2, 3])
>>> np.strings.multiply(a, i)
array(['a', 'bb', 'ccc'], dtype='<U3')
>>> np.strings.multiply(np.array(['a']), i)
array(['a', 'aa', 'aaa'], dtype='<U3')
>>> a = np.array(['a', 'b', 'c', 'd', 'e', 'f']).reshape((2, 3))
>>> np.strings.multiply(a, 3)
array([['aaa', 'bbb', 'ccc'],
       ['ddd', 'eee', 'fff']], dtype='<U3')
>>> np.strings.multiply(a, i)
array([['a', 'bb', 'ccc'],
       ['d', 'ee', 'fff']], dtype='<U3')

### Function: partition(a, sep)

**Description:** Partition each element in `a` around `sep`.

Calls :meth:`str.partition` element-wise.

For each element in `a`, split the element as the first
occurrence of `sep`, and return 3 strings containing the part
before the separator, the separator itself, and the part after
the separator. If the separator is not found, return 3 strings
containing the string itself, followed by two empty strings.

Parameters
----------
a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    Input array
sep : {str, unicode}
    Separator to split each string element in `a`.

Returns
-------
out : ndarray
    Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
    depending on input types. The output array will have an extra
    dimension with 3 elements per input element.

Examples
--------
>>> import numpy as np
>>> x = np.array(["Numpy is nice!"])
>>> np.char.partition(x, " ")
array([['Numpy', ' ', 'is nice!']], dtype='<U8')

See Also
--------
str.partition

### Function: rpartition(a, sep)

**Description:** Partition (split) each element around the right-most separator.

Calls :meth:`str.rpartition` element-wise.

For each element in `a`, split the element as the last
occurrence of `sep`, and return 3 strings containing the part
before the separator, the separator itself, and the part after
the separator. If the separator is not found, return 3 strings
containing the string itself, followed by two empty strings.

Parameters
----------
a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    Input array
sep : str or unicode
    Right-most separator to split each element in array.

Returns
-------
out : ndarray
    Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
    depending on input types. The output array will have an extra
    dimension with 3 elements per input element.

See Also
--------
str.rpartition

Examples
--------
>>> import numpy as np
>>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
>>> np.char.rpartition(a, 'A')
array([['aAaAa', 'A', ''],
   ['  a', 'A', '  '],
   ['abB', 'A', 'Bba']], dtype='<U5')

## Class: chararray

**Description:** chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
          strides=None, order=None)

Provides a convenient view on arrays of string and unicode values.

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development. Starting from numpy
   1.4, if one needs arrays of strings, it is recommended to use arrays of
   `dtype` `~numpy.object_`, `~numpy.bytes_` or `~numpy.str_`, and use
   the free functions in the `numpy.char` module for fast vectorized
   string operations.

Versus a NumPy array of dtype `~numpy.bytes_` or `~numpy.str_`, this
class adds the following functionality:

1) values automatically have whitespace removed from the end
   when indexed

2) comparison operators automatically remove whitespace from the
   end when comparing values

3) vectorized string operations are provided as methods
   (e.g. `.endswith`) and infix operators (e.g. ``"+", "*", "%"``)

chararrays should be created using `numpy.char.array` or
`numpy.char.asarray`, rather than this constructor directly.

This constructor creates the array, using `buffer` (with `offset`
and `strides`) if it is not ``None``. If `buffer` is ``None``, then
constructs a new array with `strides` in "C order", unless both
``len(shape) >= 2`` and ``order='F'``, in which case `strides`
is in "Fortran order".

Methods
-------
astype
argsort
copy
count
decode
dump
dumps
encode
endswith
expandtabs
fill
find
flatten
getfield
index
isalnum
isalpha
isdecimal
isdigit
islower
isnumeric
isspace
istitle
isupper
item
join
ljust
lower
lstrip
nonzero
put
ravel
repeat
replace
reshape
resize
rfind
rindex
rjust
rsplit
rstrip
searchsorted
setfield
setflags
sort
split
splitlines
squeeze
startswith
strip
swapaxes
swapcase
take
title
tofile
tolist
tostring
translate
transpose
upper
view
zfill

Parameters
----------
shape : tuple
    Shape of the array.
itemsize : int, optional
    Length of each array element, in number of characters. Default is 1.
unicode : bool, optional
    Are the array elements of type unicode (True) or string (False).
    Default is False.
buffer : object exposing the buffer interface or str, optional
    Memory address of the start of the array data.  Default is None,
    in which case a new array is created.
offset : int, optional
    Fixed stride displacement from the beginning of an axis?
    Default is 0. Needs to be >=0.
strides : array_like of ints, optional
    Strides for the array (see `~numpy.ndarray.strides` for
    full description). Default is None.
order : {'C', 'F'}, optional
    The order in which the array data is stored in memory: 'C' ->
    "row major" order (the default), 'F' -> "column major"
    (Fortran) order.

Examples
--------
>>> import numpy as np
>>> charar = np.char.chararray((3, 3))
>>> charar[:] = 'a'
>>> charar
chararray([[b'a', b'a', b'a'],
           [b'a', b'a', b'a'],
           [b'a', b'a', b'a']], dtype='|S1')

>>> charar = np.char.chararray(charar.shape, itemsize=5)
>>> charar[:] = 'abc'
>>> charar
chararray([[b'abc', b'abc', b'abc'],
           [b'abc', b'abc', b'abc'],
           [b'abc', b'abc', b'abc']], dtype='|S5')

### Function: array(obj, itemsize, copy, unicode, order)

**Description:** Create a `~numpy.char.chararray`.

.. note::
   This class is provided for numarray backward-compatibility.
   New code (not concerned with numarray compatibility) should use
   arrays of type `bytes_` or `str_` and use the free functions
   in :mod:`numpy.char` for fast vectorized string operations instead.

Versus a NumPy array of dtype `bytes_` or `str_`, this
class adds the following functionality:

1) values automatically have whitespace removed from the end
   when indexed

2) comparison operators automatically remove whitespace from the
   end when comparing values

3) vectorized string operations are provided as methods
   (e.g. `chararray.endswith <numpy.char.chararray.endswith>`)
   and infix operators (e.g. ``+, *, %``)

Parameters
----------
obj : array of str or unicode-like

itemsize : int, optional
    `itemsize` is the number of characters per scalar in the
    resulting array.  If `itemsize` is None, and `obj` is an
    object array or a Python list, the `itemsize` will be
    automatically determined.  If `itemsize` is provided and `obj`
    is of type str or unicode, then the `obj` string will be
    chunked into `itemsize` pieces.

copy : bool, optional
    If true (default), then the object is copied.  Otherwise, a copy
    will only be made if ``__array__`` returns a copy, if obj is a
    nested sequence, or if a copy is needed to satisfy any of the other
    requirements (`itemsize`, unicode, `order`, etc.).

unicode : bool, optional
    When true, the resulting `~numpy.char.chararray` can contain Unicode
    characters, when false only 8-bit characters.  If unicode is
    None and `obj` is one of the following:

    - a `~numpy.char.chararray`,
    - an ndarray of type :class:`str_` or :class:`bytes_`
    - a Python :class:`str` or :class:`bytes` object,

    then the unicode setting of the output array will be
    automatically determined.

order : {'C', 'F', 'A'}, optional
    Specify the order of the array.  If order is 'C' (default), then the
    array will be in C-contiguous order (last-index varies the
    fastest).  If order is 'F', then the returned array
    will be in Fortran-contiguous order (first-index varies the
    fastest).  If order is 'A', then the returned array may
    be in any order (either C-, Fortran-contiguous, or even
    discontiguous).

Examples
--------

>>> import numpy as np
>>> char_array = np.char.array(['hello', 'world', 'numpy','array'])
>>> char_array
chararray(['hello', 'world', 'numpy', 'array'], dtype='<U5')

### Function: asarray(obj, itemsize, unicode, order)

**Description:** Convert the input to a `~numpy.char.chararray`, copying the data only if
necessary.

Versus a NumPy array of dtype `bytes_` or `str_`, this
class adds the following functionality:

1) values automatically have whitespace removed from the end
   when indexed

2) comparison operators automatically remove whitespace from the
   end when comparing values

3) vectorized string operations are provided as methods
   (e.g. `chararray.endswith <numpy.char.chararray.endswith>`)
   and infix operators (e.g. ``+``, ``*``, ``%``)

Parameters
----------
obj : array of str or unicode-like

itemsize : int, optional
    `itemsize` is the number of characters per scalar in the
    resulting array.  If `itemsize` is None, and `obj` is an
    object array or a Python list, the `itemsize` will be
    automatically determined.  If `itemsize` is provided and `obj`
    is of type str or unicode, then the `obj` string will be
    chunked into `itemsize` pieces.

unicode : bool, optional
    When true, the resulting `~numpy.char.chararray` can contain Unicode
    characters, when false only 8-bit characters.  If unicode is
    None and `obj` is one of the following:

    - a `~numpy.char.chararray`,
    - an ndarray of type `str_` or `unicode_`
    - a Python str or unicode object,

    then the unicode setting of the output array will be
    automatically determined.

order : {'C', 'F'}, optional
    Specify the order of the array.  If order is 'C' (default), then the
    array will be in C-contiguous order (last-index varies the
    fastest).  If order is 'F', then the returned array
    will be in Fortran-contiguous order (first-index varies the
    fastest).

Examples
--------
>>> import numpy as np
>>> np.char.asarray(['hello', 'world'])
chararray(['hello', 'world'], dtype='<U5')

### Function: __new__(subtype, shape, itemsize, unicode, buffer, offset, strides, order)

### Function: __array_wrap__(self, arr, context, return_scalar)

### Function: __array_finalize__(self, obj)

### Function: __getitem__(self, obj)

### Function: __eq__(self, other)

**Description:** Return (self == other) element-wise.

See Also
--------
equal

### Function: __ne__(self, other)

**Description:** Return (self != other) element-wise.

See Also
--------
not_equal

### Function: __ge__(self, other)

**Description:** Return (self >= other) element-wise.

See Also
--------
greater_equal

### Function: __le__(self, other)

**Description:** Return (self <= other) element-wise.

See Also
--------
less_equal

### Function: __gt__(self, other)

**Description:** Return (self > other) element-wise.

See Also
--------
greater

### Function: __lt__(self, other)

**Description:** Return (self < other) element-wise.

See Also
--------
less

### Function: __add__(self, other)

**Description:** Return (self + other), that is string concatenation,
element-wise for a pair of array_likes of str or unicode.

See Also
--------
add

### Function: __radd__(self, other)

**Description:** Return (other + self), that is string concatenation,
element-wise for a pair of array_likes of `bytes_` or `str_`.

See Also
--------
add

### Function: __mul__(self, i)

**Description:** Return (self * i), that is string multiple concatenation,
element-wise.

See Also
--------
multiply

### Function: __rmul__(self, i)

**Description:** Return (self * i), that is string multiple concatenation,
element-wise.

See Also
--------
multiply

### Function: __mod__(self, i)

**Description:** Return (self % i), that is pre-Python 2.6 string formatting
(interpolation), element-wise for a pair of array_likes of `bytes_`
or `str_`.

See Also
--------
mod

### Function: __rmod__(self, other)

### Function: argsort(self, axis, kind, order)

**Description:** Return the indices that sort the array lexicographically.

For full documentation see `numpy.argsort`, for which this method is
in fact merely a "thin wrapper."

Examples
--------
>>> c = np.array(['a1b c', '1b ca', 'b ca1', 'Ca1b'], 'S5')
>>> c = c.view(np.char.chararray); c
chararray(['a1b c', '1b ca', 'b ca1', 'Ca1b'],
      dtype='|S5')
>>> c[c.argsort()]
chararray(['1b ca', 'Ca1b', 'a1b c', 'b ca1'],
      dtype='|S5')

### Function: capitalize(self)

**Description:** Return a copy of `self` with only the first character of each element
capitalized.

See Also
--------
char.capitalize

### Function: center(self, width, fillchar)

**Description:** Return a copy of `self` with its elements centered in a
string of length `width`.

See Also
--------
center

### Function: count(self, sub, start, end)

**Description:** Returns an array with the number of non-overlapping occurrences of
substring `sub` in the range [`start`, `end`].

See Also
--------
char.count

### Function: decode(self, encoding, errors)

**Description:** Calls ``bytes.decode`` element-wise.

See Also
--------
char.decode

### Function: encode(self, encoding, errors)

**Description:** Calls :meth:`str.encode` element-wise.

See Also
--------
char.encode

### Function: endswith(self, suffix, start, end)

**Description:** Returns a boolean array which is `True` where the string element
in `self` ends with `suffix`, otherwise `False`.

See Also
--------
char.endswith

### Function: expandtabs(self, tabsize)

**Description:** Return a copy of each string element where all tab characters are
replaced by one or more spaces.

See Also
--------
char.expandtabs

### Function: find(self, sub, start, end)

**Description:** For each element, return the lowest index in the string where
substring `sub` is found.

See Also
--------
char.find

### Function: index(self, sub, start, end)

**Description:** Like `find`, but raises :exc:`ValueError` when the substring is not
found.

See Also
--------
char.index

### Function: isalnum(self)

**Description:** Returns true for each element if all characters in the string
are alphanumeric and there is at least one character, false
otherwise.

See Also
--------
char.isalnum

### Function: isalpha(self)

**Description:** Returns true for each element if all characters in the string
are alphabetic and there is at least one character, false
otherwise.

See Also
--------
char.isalpha

### Function: isdigit(self)

**Description:** Returns true for each element if all characters in the string are
digits and there is at least one character, false otherwise.

See Also
--------
char.isdigit

### Function: islower(self)

**Description:** Returns true for each element if all cased characters in the
string are lowercase and there is at least one cased character,
false otherwise.

See Also
--------
char.islower

### Function: isspace(self)

**Description:** Returns true for each element if there are only whitespace
characters in the string and there is at least one character,
false otherwise.

See Also
--------
char.isspace

### Function: istitle(self)

**Description:** Returns true for each element if the element is a titlecased
string and there is at least one character, false otherwise.

See Also
--------
char.istitle

### Function: isupper(self)

**Description:** Returns true for each element if all cased characters in the
string are uppercase and there is at least one character, false
otherwise.

See Also
--------
char.isupper

### Function: join(self, seq)

**Description:** Return a string which is the concatenation of the strings in the
sequence `seq`.

See Also
--------
char.join

### Function: ljust(self, width, fillchar)

**Description:** Return an array with the elements of `self` left-justified in a
string of length `width`.

See Also
--------
char.ljust

### Function: lower(self)

**Description:** Return an array with the elements of `self` converted to
lowercase.

See Also
--------
char.lower

### Function: lstrip(self, chars)

**Description:** For each element in `self`, return a copy with the leading characters
removed.

See Also
--------
char.lstrip

### Function: partition(self, sep)

**Description:** Partition each element in `self` around `sep`.

See Also
--------
partition

### Function: replace(self, old, new, count)

**Description:** For each element in `self`, return a copy of the string with all
occurrences of substring `old` replaced by `new`.

See Also
--------
char.replace

### Function: rfind(self, sub, start, end)

**Description:** For each element in `self`, return the highest index in the string
where substring `sub` is found, such that `sub` is contained
within [`start`, `end`].

See Also
--------
char.rfind

### Function: rindex(self, sub, start, end)

**Description:** Like `rfind`, but raises :exc:`ValueError` when the substring `sub` is
not found.

See Also
--------
char.rindex

### Function: rjust(self, width, fillchar)

**Description:** Return an array with the elements of `self`
right-justified in a string of length `width`.

See Also
--------
char.rjust

### Function: rpartition(self, sep)

**Description:** Partition each element in `self` around `sep`.

See Also
--------
rpartition

### Function: rsplit(self, sep, maxsplit)

**Description:** For each element in `self`, return a list of the words in
the string, using `sep` as the delimiter string.

See Also
--------
char.rsplit

### Function: rstrip(self, chars)

**Description:** For each element in `self`, return a copy with the trailing
characters removed.

See Also
--------
char.rstrip

### Function: split(self, sep, maxsplit)

**Description:** For each element in `self`, return a list of the words in the
string, using `sep` as the delimiter string.

See Also
--------
char.split

### Function: splitlines(self, keepends)

**Description:** For each element in `self`, return a list of the lines in the
element, breaking at line boundaries.

See Also
--------
char.splitlines

### Function: startswith(self, prefix, start, end)

**Description:** Returns a boolean array which is `True` where the string element
in `self` starts with `prefix`, otherwise `False`.

See Also
--------
char.startswith

### Function: strip(self, chars)

**Description:** For each element in `self`, return a copy with the leading and
trailing characters removed.

See Also
--------
char.strip

### Function: swapcase(self)

**Description:** For each element in `self`, return a copy of the string with
uppercase characters converted to lowercase and vice versa.

See Also
--------
char.swapcase

### Function: title(self)

**Description:** For each element in `self`, return a titlecased version of the
string: words start with uppercase characters, all remaining cased
characters are lowercase.

See Also
--------
char.title

### Function: translate(self, table, deletechars)

**Description:** For each element in `self`, return a copy of the string where
all characters occurring in the optional argument
`deletechars` are removed, and the remaining characters have
been mapped through the given translation table.

See Also
--------
char.translate

### Function: upper(self)

**Description:** Return an array with the elements of `self` converted to
uppercase.

See Also
--------
char.upper

### Function: zfill(self, width)

**Description:** Return the numeric string left-filled with zeros in a string of
length `width`.

See Also
--------
char.zfill

### Function: isnumeric(self)

**Description:** For each element in `self`, return True if there are only
numeric characters in the element.

See Also
--------
char.isnumeric

### Function: isdecimal(self)

**Description:** For each element in `self`, return True if there are only
decimal characters in the element.

See Also
--------
char.isdecimal
