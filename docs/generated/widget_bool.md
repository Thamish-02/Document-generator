## AI Summary

A file named widget_bool.py.


## Class: CheckboxStyle

**Description:** Checkbox widget style.

## Class: ToggleButtonStyle

**Description:** ToggleButton widget style.

## Class: _Bool

**Description:** A base class for creating widgets that represent booleans.

## Class: Checkbox

**Description:** Displays a boolean `value` in the form of a checkbox.

Parameters
----------
value : {True,False}
    value of the checkbox: True-checked, False-unchecked
description : str
    description displayed next to the checkbox
indent : {True,False}
    indent the control to align with other controls with a description. The style.description_width attribute controls this width for consistence with other controls.

## Class: ToggleButton

**Description:** Displays a boolean `value` in the form of a toggle button.

Parameters
----------
value : {True,False}
    value of the toggle button: True-pressed, False-unpressed
description : str
    description displayed on the button
icon: str
    font-awesome icon name
style: instance of DescriptionStyle
    styling customizations
button_style: enum
    button predefined styling

## Class: Valid

**Description:** Displays a boolean `value` in the form of a green check (True / valid)
or a red cross (False / invalid).

Parameters
----------
value: {True,False}
    value of the Valid widget

### Function: __init__(self, value)
