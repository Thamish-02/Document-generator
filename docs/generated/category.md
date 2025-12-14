## AI Summary

A file named category.py.


## Class: StrCategoryConverter

## Class: StrCategoryLocator

**Description:** Tick at every integer mapping of the string data.

## Class: StrCategoryFormatter

**Description:** String representation of the data at every tick.

## Class: UnitData

### Function: convert(value, unit, axis)

**Description:** Convert strings in *value* to floats using mapping information stored
in the *unit* object.

Parameters
----------
value : str or iterable
    Value or list of values to be converted.
unit : `.UnitData`
    An object mapping strings to integers.
axis : `~matplotlib.axis.Axis`
    The axis on which the converted value is plotted.

    .. note:: *axis* is unused.

Returns
-------
float or `~numpy.ndarray` of float

### Function: axisinfo(unit, axis)

**Description:** Set the default axis ticks and labels.

Parameters
----------
unit : `.UnitData`
    object string unit information for value
axis : `~matplotlib.axis.Axis`
    axis for which information is being set

    .. note:: *axis* is not used

Returns
-------
`~matplotlib.units.AxisInfo`
    Information to support default tick labeling

### Function: default_units(data, axis)

**Description:** Set and update the `~matplotlib.axis.Axis` units.

Parameters
----------
data : str or iterable of str
axis : `~matplotlib.axis.Axis`
    axis on which the data is plotted

Returns
-------
`.UnitData`
    object storing string to integer mapping

### Function: _validate_unit(unit)

### Function: __init__(self, units_mapping)

**Description:** Parameters
----------
units_mapping : dict
    Mapping of category names (str) to indices (int).

### Function: __call__(self)

### Function: tick_values(self, vmin, vmax)

### Function: __init__(self, units_mapping)

**Description:** Parameters
----------
units_mapping : dict
    Mapping of category names (str) to indices (int).

### Function: __call__(self, x, pos)

### Function: format_ticks(self, values)

### Function: _text(value)

**Description:** Convert text values into utf-8 or ascii strings.

### Function: __init__(self, data)

**Description:** Create mapping between unique categorical values and integer ids.

Parameters
----------
data : iterable
    sequence of string values

### Function: _str_is_convertible(val)

**Description:** Helper method to check whether a string can be parsed as float or date.

### Function: update(self, data)

**Description:** Map new values to integer identifiers.

Parameters
----------
data : iterable of str or bytes

Raises
------
TypeError
    If elements in *data* are neither str nor bytes.
