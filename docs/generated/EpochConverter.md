## AI Summary

A file named EpochConverter.py.


## Class: EpochConverter

**Description:** Provides Matplotlib conversion functionality for Monte Epoch and Duration
classes.

### Function: axisinfo(unit, axis)

### Function: float2epoch(value, unit)

**Description:** Convert a Matplotlib floating-point date into an Epoch of the specified
units.

= INPUT VARIABLES
- value     The Matplotlib floating-point date.
- unit      The unit system to use for the Epoch.

= RETURN VALUE
- Returns the value converted to an Epoch in the specified time system.

### Function: epoch2float(value, unit)

**Description:** Convert an Epoch value to a float suitable for plotting as a python
datetime object.

= INPUT VARIABLES
- value    An Epoch or list of Epochs that need to be converted.
- unit     The units to use for an axis with Epoch data.

= RETURN VALUE
- Returns the value parameter converted to floats.

### Function: duration2float(value)

**Description:** Convert a Duration value to a float suitable for plotting as a python
datetime object.

= INPUT VARIABLES
- value    A Duration or list of Durations that need to be converted.

= RETURN VALUE
- Returns the value parameter converted to floats.

### Function: convert(value, unit, axis)

### Function: default_units(value, axis)
