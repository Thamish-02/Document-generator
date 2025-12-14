## AI Summary

A file named widget_date.py.


## Class: DatePicker

**Description:** Display a widget for picking dates.

Parameters
----------

value: datetime.date
    The current value of the widget.

disabled: bool
    Whether to disable user changes.

Examples
--------

>>> import datetime
>>> import ipywidgets as widgets
>>> date_pick = widgets.DatePicker()
>>> date_pick.value = datetime.date(2019, 7, 9)

### Function: _validate_value(self, proposal)

**Description:** Cap and floor value

### Function: _validate_min(self, proposal)

**Description:** Enforce min <= value <= max

### Function: _validate_max(self, proposal)

**Description:** Enforce min <= value <= max
