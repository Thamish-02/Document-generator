## AI Summary

A file named widget_float.py.


## Class: _Float

## Class: _BoundedFloat

## Class: _BoundedLogFloat

## Class: FloatText

**Description:** Displays a float value within a textbox. For a textbox in
which the value must be within a specific range, use BoundedFloatText.

Parameters
----------
value : float
    value displayed
step : float
    step of the increment (if None, any step is allowed)
description : str
    description displayed next to the text box

## Class: BoundedFloatText

**Description:** Displays a float value within a textbox. Value must be within the range specified.

For a textbox in which the value doesn't need to be within a specific range, use FloatText.

Parameters
----------
value : float
    value displayed
min : float
    minimal value of the range of possible values displayed
max : float
    maximal value of the range of possible values displayed
step : float
    step of the increment (if None, any step is allowed)
description : str
    description displayed next to the textbox

## Class: FloatSlider

**Description:** Slider/trackbar of floating values with the specified range.

Parameters
----------
value : float
    position of the slider
min : float
    minimal position of the slider
max : float
    maximal position of the slider
step : float
    step of the trackbar
description : str
    name of the slider
orientation : {'horizontal', 'vertical'}
    default is 'horizontal', orientation of the slider
readout : {True, False}
    default is True, display the current value of the slider next to it
behavior : str
    slider handle and connector dragging behavior. Default is 'drag-tap'.
readout_format : str
    default is '.2f', specifier for the format function used to represent
    slider value for human consumption, modeled after Python 3's format
    specification mini-language (PEP 3101).

## Class: FloatLogSlider

**Description:** Slider/trackbar of logarithmic floating values with the specified range.

Parameters
----------
value : float
    position of the slider
base : float
    base of the logarithmic scale. Default is 10
min : float
    minimal position of the slider in log scale, i.e., actual minimum is base ** min
max : float
    maximal position of the slider in log scale, i.e., actual maximum is base ** max
step : float
    step of the trackbar, denotes steps for the exponent, not the actual value
description : str
    name of the slider
orientation : {'horizontal', 'vertical'}
    default is 'horizontal', orientation of the slider
readout : {True, False}
    default is True, display the current value of the slider next to it
behavior : str
    slider handle and connector dragging behavior. Default is 'drag-tap'.
readout_format : str
    default is '.3g', specifier for the format function used to represent
    slider value for human consumption, modeled after Python 3's format
    specification mini-language (PEP 3101).

## Class: FloatProgress

**Description:** Displays a progress bar.

Parameters
-----------
value : float
    position within the range of the progress bar
min : float
    minimal position of the slider
max : float
    maximal position of the slider
description : str
    name of the progress bar
orientation : {'horizontal', 'vertical'}
    default is 'horizontal', orientation of the progress bar
bar_style: {'success', 'info', 'warning', 'danger', ''}
    color of the progress bar, default is '' (blue)
    colors are: 'success'-green, 'info'-light blue, 'warning'-orange, 'danger'-red

## Class: _FloatRange

## Class: _BoundedFloatRange

## Class: FloatRangeSlider

**Description:** Slider/trackbar that represents a pair of floats bounded by minimum and maximum value.

Parameters
----------
value : float tuple
    range of the slider displayed
min : float
    minimal position of the slider
max : float
    maximal position of the slider
step : float
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

### Function: _validate_value(self, proposal)

**Description:** Cap and floor value

### Function: _validate_min(self, proposal)

**Description:** Enforce min <= value <= max

### Function: _validate_max(self, proposal)

**Description:** Enforce min <= value <= max

### Function: _validate_value(self, proposal)

**Description:** Cap and floor value

### Function: _validate_min(self, proposal)

**Description:** Enforce base ** min <= value <= base ** max

### Function: _validate_max(self, proposal)

**Description:** Enforce base ** min <= value <= base ** max

### Function: lower(self)

### Function: lower(self, lower)

### Function: upper(self)

### Function: upper(self, upper)

### Function: _validate_value(self, proposal)

### Function: __init__(self)

### Function: _validate_bounds(self, proposal)

### Function: _validate_value(self, proposal)
