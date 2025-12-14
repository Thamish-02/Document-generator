## AI Summary

A file named widget_int.py.


### Function: _int_doc(cls)

**Description:** Add int docstring template to class init.

### Function: _bounded_int_doc(cls)

**Description:** Add bounded int docstring template to class init.

## Class: _Int

**Description:** Base class for widgets that represent an integer.

## Class: _BoundedInt

**Description:** Base class for widgets that represent an integer bounded from above and below.
    

## Class: IntText

**Description:** Textbox widget that represents an integer.

## Class: BoundedIntText

**Description:** Textbox widget that represents an integer bounded from above and below.
    

## Class: SliderStyle

**Description:** Button style widget.

## Class: IntSlider

**Description:** Slider widget that represents an integer bounded from above and below.
    

## Class: ProgressStyle

**Description:** Button style widget.

## Class: IntProgress

**Description:** Progress bar that represents an integer bounded from above and below.
    

## Class: _IntRange

## Class: Play

**Description:** Play/repeat buttons to step through values automatically, and optionally loop.
    

## Class: _BoundedIntRange

## Class: IntRangeSlider

**Description:** Slider/trackbar that represents a pair of ints bounded by minimum and maximum value.

Parameters
----------
value : int tuple
    The pair (`lower`, `upper`) of integers
min : int
    The lowest allowed value for `lower`
max : int
    The highest allowed value for `upper`
step : int
    step of the trackbar
description : str
    name of the slider
orientation : {'horizontal', 'vertical'}
    default is 'horizontal'
readout : {True, False}
    default is True, display the current value of the slider next to it
behavior : str
    slider handle and connector dragging behavior. Default is 'drag-tap'.
readout_format : str
    default is '.2f', specifier for the format function used to represent
    slider value for human consumption, modeled after Python 3's format
    specification mini-language (PEP 3101).

### Function: __init__(self, value)

### Function: __init__(self, value, min, max, step)

### Function: __init__(self, value)

### Function: __init__(self, value, min, max, step)

### Function: _validate_value(self, proposal)

**Description:** Cap and floor value

### Function: _validate_min(self, proposal)

**Description:** Enforce min <= value <= max

### Function: _validate_max(self, proposal)

**Description:** Enforce min <= value <= max

### Function: lower(self)

### Function: lower(self, lower)

### Function: upper(self)

### Function: upper(self, upper)

### Function: _validate_value(self, proposal)

### Function: __init__(self)

### Function: _validate_bounds(self, proposal)

### Function: _validate_value(self, proposal)
