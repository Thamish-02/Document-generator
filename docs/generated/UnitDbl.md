## AI Summary

A file named UnitDbl.py.


## Class: UnitDbl

**Description:** Class UnitDbl in development.

### Function: __init__(self, value, units)

**Description:** Create a new UnitDbl object.

Units are internally converted to km, rad, and sec.  The only
valid inputs for units are [m, km, mile, rad, deg, sec, min, hour].

The field UnitDbl.value will contain the converted value.  Use
the convert() method to get a specific type of units back.

= ERROR CONDITIONS
- If the input units are not in the allowed list, an error is thrown.

= INPUT VARIABLES
- value     The numeric value of the UnitDbl.
- units     The string name of the units the value is in.

### Function: convert(self, units)

**Description:** Convert the UnitDbl to a specific set of units.

= ERROR CONDITIONS
- If the input units are not in the allowed list, an error is thrown.

= INPUT VARIABLES
- units     The string name of the units to convert to.

= RETURN VALUE
- Returns the value of the UnitDbl in the requested units as a floating
  point number.

### Function: __abs__(self)

**Description:** Return the absolute value of this UnitDbl.

### Function: __neg__(self)

**Description:** Return the negative value of this UnitDbl.

### Function: __bool__(self)

**Description:** Return the truth value of a UnitDbl.

### Function: _cmp(self, op, rhs)

**Description:** Check that *self* and *rhs* share units; compare them using *op*.

### Function: _binop_unit_unit(self, op, rhs)

**Description:** Check that *self* and *rhs* share units; combine them using *op*.

### Function: _binop_unit_scalar(self, op, scalar)

**Description:** Combine *self* and *scalar* using *op*.

### Function: __str__(self)

**Description:** Print the UnitDbl.

### Function: __repr__(self)

**Description:** Print the UnitDbl.

### Function: type(self)

**Description:** Return the type of UnitDbl data.

### Function: range(start, stop, step)

**Description:** Generate a range of UnitDbl objects.

Similar to the Python range() method.  Returns the range [
start, stop) at the requested step.  Each element will be a
UnitDbl object.

= INPUT VARIABLES
- start     The starting value of the range.
- stop      The stop value of the range.
- step      Optional step to use.  If set to None, then a UnitDbl of
              value 1 w/ the units of the start is used.

= RETURN VALUE
- Returns a list containing the requested UnitDbl values.

### Function: checkSameUnits(self, rhs, func)

**Description:** Check to see if units are the same.

= ERROR CONDITIONS
- If the units of the rhs UnitDbl are not the same as our units,
  an error is thrown.

= INPUT VARIABLES
- rhs     The UnitDbl to check for the same units
- func    The name of the function doing the check.
