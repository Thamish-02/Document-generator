## AI Summary

A file named units.py.


## Class: ConversionError

### Function: _is_natively_supported(x)

**Description:** Return whether *x* is of a type that Matplotlib natively supports or an
array of objects of such types.

## Class: AxisInfo

**Description:** Information to support default axis labeling, tick labeling, and limits.

An instance of this class must be returned by
`ConversionInterface.axisinfo`.

## Class: ConversionInterface

**Description:** The minimal interface for a converter to take custom data types (or
sequences) and convert them to values Matplotlib can use.

## Class: DecimalConverter

**Description:** Converter for decimal.Decimal data to float.

## Class: Registry

**Description:** Register types with conversion interface.

### Function: __init__(self, majloc, minloc, majfmt, minfmt, label, default_limits)

**Description:** Parameters
----------
majloc, minloc : Locator, optional
    Tick locators for the major and minor ticks.
majfmt, minfmt : Formatter, optional
    Tick formatters for the major and minor ticks.
label : str, optional
    The default axis label.
default_limits : optional
    The default min and max limits of the axis if no data has
    been plotted.

Notes
-----
If any of the above are ``None``, the axis will simply use the
default value.

### Function: axisinfo(unit, axis)

**Description:** Return an `.AxisInfo` for the axis with the specified units.

### Function: default_units(x, axis)

**Description:** Return the default unit for *x* or ``None`` for the given axis.

### Function: convert(obj, unit, axis)

**Description:** Convert *obj* using *unit* for the specified *axis*.

If *obj* is a sequence, return the converted sequence.  The output must
be a sequence of scalars that can be used by the numpy array layer.

### Function: convert(value, unit, axis)

**Description:** Convert Decimals to floats.

The *unit* and *axis* arguments are not used.

Parameters
----------
value : decimal.Decimal or iterable
    Decimal or list of Decimal need to be converted

### Function: get_converter(self, x)

**Description:** Get the converter interface instance for *x*, or None.
