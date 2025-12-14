## AI Summary

A file named getlimits.py.


### Function: _fr0(a)

**Description:** fix rank-0 --> rank-1

### Function: _fr1(a)

**Description:** fix rank > 0 --> rank-0

## Class: MachArLike

**Description:** Object to simulate MachAr instance 

### Function: _register_type(machar, bytepat)

### Function: _register_known_types()

### Function: _get_machar(ftype)

**Description:** Get MachAr instance or MachAr-like instance

Get parameters for floating point type, by first trying signatures of
various known floating point types, then, if none match, attempting to
identify parameters by analysis.

Parameters
----------
ftype : class
    Numpy floating point type class (e.g. ``np.float64``)

Returns
-------
ma_like : instance of :class:`MachAr` or :class:`MachArLike`
    Object giving floating point parameters for `ftype`.

Warns
-----
UserWarning
    If the binary signature of the float type is not in the dictionary of
    known float types.

### Function: _discovered_machar(ftype)

**Description:** Create MachAr instance with found information on float types

TODO: MachAr should be retired completely ideally.  We currently only
      ever use it system with broken longdouble (valgrind, WSL).

## Class: finfo

**Description:** finfo(dtype)

Machine limits for floating point types.

Attributes
----------
bits : int
    The number of bits occupied by the type.
dtype : dtype
    Returns the dtype for which `finfo` returns information. For complex
    input, the returned dtype is the associated ``float*`` dtype for its
    real and complex components.
eps : float
    The difference between 1.0 and the next smallest representable float
    larger than 1.0. For example, for 64-bit binary floats in the IEEE-754
    standard, ``eps = 2**-52``, approximately 2.22e-16.
epsneg : float
    The difference between 1.0 and the next smallest representable float
    less than 1.0. For example, for 64-bit binary floats in the IEEE-754
    standard, ``epsneg = 2**-53``, approximately 1.11e-16.
iexp : int
    The number of bits in the exponent portion of the floating point
    representation.
machep : int
    The exponent that yields `eps`.
max : floating point number of the appropriate type
    The largest representable number.
maxexp : int
    The smallest positive power of the base (2) that causes overflow.
min : floating point number of the appropriate type
    The smallest representable number, typically ``-max``.
minexp : int
    The most negative power of the base (2) consistent with there
    being no leading 0's in the mantissa.
negep : int
    The exponent that yields `epsneg`.
nexp : int
    The number of bits in the exponent including its sign and bias.
nmant : int
    The number of bits in the mantissa.
precision : int
    The approximate number of decimal digits to which this kind of
    float is precise.
resolution : floating point number of the appropriate type
    The approximate decimal resolution of this type, i.e.,
    ``10**-precision``.
tiny : float
    An alias for `smallest_normal`, kept for backwards compatibility.
smallest_normal : float
    The smallest positive floating point number with 1 as leading bit in
    the mantissa following IEEE-754 (see Notes).
smallest_subnormal : float
    The smallest positive floating point number with 0 as leading bit in
    the mantissa following IEEE-754.

Parameters
----------
dtype : float, dtype, or instance
    Kind of floating point or complex floating point
    data-type about which to get information.

See Also
--------
iinfo : The equivalent for integer data types.
spacing : The distance between a value and the nearest adjacent number
nextafter : The next floating point value after x1 towards x2

Notes
-----
For developers of NumPy: do not instantiate this at the module level.
The initial calculation of these parameters is expensive and negatively
impacts import times.  These objects are cached, so calling ``finfo()``
repeatedly inside your functions is not a problem.

Note that ``smallest_normal`` is not actually the smallest positive
representable value in a NumPy floating point type. As in the IEEE-754
standard [1]_, NumPy floating point types make use of subnormal numbers to
fill the gap between 0 and ``smallest_normal``. However, subnormal numbers
may have significantly reduced precision [2]_.

This function can also be used for complex data types as well. If used,
the output will be the same as the corresponding real float type
(e.g. numpy.finfo(numpy.csingle) is the same as numpy.finfo(numpy.single)).
However, the output is true for the real and imaginary components.

References
----------
.. [1] IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2008,
       pp.1-70, 2008, https://doi.org/10.1109/IEEESTD.2008.4610935
.. [2] Wikipedia, "Denormal Numbers",
       https://en.wikipedia.org/wiki/Denormal_number

Examples
--------
>>> import numpy as np
>>> np.finfo(np.float64).dtype
dtype('float64')
>>> np.finfo(np.complex64).dtype
dtype('float32')

## Class: iinfo

**Description:** iinfo(type)

Machine limits for integer types.

Attributes
----------
bits : int
    The number of bits occupied by the type.
dtype : dtype
    Returns the dtype for which `iinfo` returns information.
min : int
    The smallest integer expressible by the type.
max : int
    The largest integer expressible by the type.

Parameters
----------
int_type : integer type, dtype, or instance
    The kind of integer data type to get information about.

See Also
--------
finfo : The equivalent for floating point data types.

Examples
--------
With types:

>>> import numpy as np
>>> ii16 = np.iinfo(np.int16)
>>> ii16.min
-32768
>>> ii16.max
32767
>>> ii32 = np.iinfo(np.int32)
>>> ii32.min
-2147483648
>>> ii32.max
2147483647

With instances:

>>> ii32 = np.iinfo(np.int32(10))
>>> ii32.min
-2147483648
>>> ii32.max
2147483647

### Function: __init__(self, ftype)

### Function: smallest_subnormal(self)

**Description:** Return the value for the smallest subnormal.

Returns
-------
smallest_subnormal : float
    value for the smallest subnormal.

Warns
-----
UserWarning
    If the calculated value for the smallest subnormal is zero.

### Function: _str_smallest_subnormal(self)

**Description:** Return the string representation of the smallest subnormal.

### Function: _float_to_float(self, value)

**Description:** Converts float to float.

Parameters
----------
value : float
    value to be converted.

### Function: _float_conv(self, value)

**Description:** Converts float to conv.

Parameters
----------
value : float
    value to be converted.

### Function: _float_to_str(self, value)

**Description:** Converts float to str.

Parameters
----------
value : float
    value to be converted.

### Function: __new__(cls, dtype)

### Function: _init(self, dtype)

### Function: __str__(self)

### Function: __repr__(self)

### Function: smallest_normal(self)

**Description:** Return the value for the smallest normal.

Returns
-------
smallest_normal : float
    Value for the smallest normal.

Warns
-----
UserWarning
    If the calculated value for the smallest normal is requested for
    double-double.

### Function: tiny(self)

**Description:** Return the value for tiny, alias of smallest_normal.

Returns
-------
tiny : float
    Value for the smallest normal, alias of smallest_normal.

Warns
-----
UserWarning
    If the calculated value for the smallest normal is requested for
    double-double.

### Function: __init__(self, int_type)

### Function: min(self)

**Description:** Minimum value of given dtype.

### Function: max(self)

**Description:** Maximum value of given dtype.

### Function: __str__(self)

**Description:** String representation.

### Function: __repr__(self)
