## AI Summary

A file named widget_selection.py.


### Function: _exhaust_iterable(x)

**Description:** Exhaust any non-mapping iterable into a tuple

### Function: _make_options(x)

**Description:** Standardize the options tuple format.

The returned tuple should be in the format (('label', value), ('label', value), ...).

The input can be
* an iterable of (label, value) pairs
* an iterable of values, and labels will be generated
* a Mapping between labels and values

### Function: findvalue(array, value, compare)

**Description:** A function that uses the compare function to return a value from the list.

## Class: _Selection

**Description:** Base class for Selection widgets

``options`` can be specified as a list of values or a list of (label, value)
tuples. The labels are the strings that will be displayed in the UI,
representing the actual Python choices, and should be unique.
If labels are not specified, they are generated from the values.

When programmatically setting the value, a reverse lookup is performed
among the options to check that the value is valid. The reverse lookup uses
the equality operator by default, but another predicate may be provided via
the ``equals`` keyword argument. For example, when dealing with numpy arrays,
one may set equals=np.array_equal.

## Class: _MultipleSelection

**Description:** Base class for multiple Selection widgets

``options`` can be specified as a list of values, list of (label, value)
tuples, or a dict of {label: value}. The labels are the strings that will be
displayed in the UI, representing the actual Python choices, and should be
unique. If labels are not specified, they are generated from the values.

When programmatically setting the value, a reverse lookup is performed
among the options to check that the value is valid. The reverse lookup uses
the equality operator by default, but another predicate may be provided via
the ``equals`` keyword argument. For example, when dealing with numpy arrays,
one may set equals=np.array_equal.

## Class: ToggleButtonsStyle

**Description:** Button style widget.

Parameters
----------
button_width: str
    The width of each button. This should be a valid CSS
    width, e.g. '10px' or '5em'.

font_weight: str
    The text font weight of each button, This should be a valid CSS font
    weight unit, for example 'bold' or '600'

## Class: ToggleButtons

**Description:** Group of toggle buttons that represent an enumeration.

Only one toggle button can be toggled at any point in time.

Parameters
----------
{selection_params}

tooltips: list
    Tooltip for each button. If specified, must be the
    same length as `options`.

icons: list
    Icons to show on the buttons. This must be the name
    of a font-awesome icon. See `http://fontawesome.io/icons/`
    for a list of icons.

button_style: str
    One of 'primary', 'success', 'info', 'warning' or
    'danger'. Applies a predefined style to every button.

style: ToggleButtonsStyle
    Style parameters for the buttons.

## Class: Dropdown

**Description:** Allows you to select a single item from a dropdown.

Parameters
----------
{selection_params}

## Class: RadioButtons

**Description:** Group of radio buttons that represent an enumeration.

Only one radio button can be toggled at any point in time.

Parameters
----------
{selection_params}

## Class: Select

**Description:** Listbox that only allows one item to be selected at any given time.

Parameters
----------
{selection_params}

rows: int
    The number of rows to display in the widget.

## Class: SelectMultiple

**Description:** Listbox that allows many items to be selected at any given time.

The ``value``, ``label`` and ``index`` attributes are all iterables.

Parameters
----------
{multiple_selection_params}

rows: int
    The number of rows to display in the widget.

## Class: _SelectionNonempty

**Description:** Selection that is guaranteed to have a value selected.

## Class: _MultipleSelectionNonempty

**Description:** Selection that is guaranteed to have an option available.

## Class: SelectionSlider

**Description:** Slider to select a single item from a list or dictionary.

Parameters
----------
{selection_params}

{slider_params}

## Class: SelectionRangeSlider

**Description:** Slider to select multiple contiguous items from a list.

The index, value, and label attributes contain the start and end of
the selection range, not all items in the range.

Parameters
----------
{multiple_selection_params}

{slider_params}

### Function: __init__(self)

### Function: _validate_options(self, proposal)

### Function: _propagate_options(self, change)

**Description:** Set the values and labels, and select the first option if we aren't initializing

### Function: _validate_index(self, proposal)

### Function: _propagate_index(self, change)

**Description:** Propagate changes in index to the value and label properties

### Function: _validate_value(self, proposal)

### Function: _propagate_value(self, change)

### Function: _validate_label(self, proposal)

### Function: _propagate_label(self, change)

### Function: _repr_keys(self)

### Function: __init__(self)

### Function: _validate_options(self, proposal)

### Function: _propagate_options(self, change)

**Description:** Unselect any option

### Function: _validate_index(self, proposal)

**Description:** Check the range of each proposed index.

### Function: _propagate_index(self, change)

**Description:** Propagate changes in index to the value and label properties

### Function: _validate_value(self, proposal)

**Description:** Replace all values with the actual objects in the options list

### Function: _propagate_value(self, change)

### Function: _validate_label(self, proposal)

### Function: _propagate_label(self, change)

### Function: _repr_keys(self)

### Function: __init__(self)

### Function: _validate_options(self, proposal)

### Function: _validate_index(self, proposal)

### Function: __init__(self)

### Function: _validate_options(self, proposal)

### Function: _propagate_options(self, change)

**Description:** Select the first range

### Function: _validate_index(self, proposal)

**Description:** Make sure we have two indices and check the range of each proposed index.
