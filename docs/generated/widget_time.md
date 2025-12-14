## AI Summary

A file named widget_time.py.


## Class: TimePicker

**Description:** Display a widget for picking times.

Parameters
----------

value: datetime.time
    The current value of the widget.

disabled: bool
    Whether to disable user changes.

min: datetime.time
    The lower allowed time bound

max: datetime.time
    The upper allowed time bound

step: float | 'any'
    The time step to use for the picker, in seconds, or "any"

Examples
--------

>>> import datetime
>>> import ipydatetime
>>> time_pick = ipydatetime.TimePicker()
>>> time_pick.value = datetime.time(12, 34, 3)

### Function: _validate_value(self, proposal)

**Description:** Cap and floor value

### Function: _validate_min(self, proposal)

**Description:** Enforce min <= value <= max

### Function: _validate_max(self, proposal)

**Description:** Enforce min <= value <= max
