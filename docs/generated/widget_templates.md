## AI Summary

A file named widget_templates.py.


## Class: LayoutProperties

**Description:** Mixin class for layout templates

This class handles mainly style attributes (height, grid_gap etc.)

Parameters
----------

{style_params}


Note
----

This class is only meant to be used in inheritance as mixin with other
classes. It will not work, unless `self.layout` attribute is defined.

## Class: AppLayout

**Description:** Define an application like layout of widgets.

Parameters
----------

header: instance of Widget
left_sidebar: instance of Widget
center: instance of Widget
right_sidebar: instance of Widget
footer: instance of Widget
    widgets to fill the positions in the layout

merge: bool
    flag to say whether the empty positions should be automatically merged

pane_widths: list of numbers/strings
    the fraction of the total layout width each of the central panes should occupy
    (left_sidebar,
    center, right_sidebar)

pane_heights: list of numbers/strings
    the fraction of the width the vertical space that the panes should occupy
     (left_sidebar, center, right_sidebar)

{style_params}

Examples
--------

## Class: GridspecLayout

**Description:** Define a N by M grid layout

Parameters
----------

n_rows : int
    number of rows in the grid

n_columns : int
    number of columns in the grid

{style_params}

Examples
--------

>>> from ipywidgets import GridspecLayout, Button, Layout
>>> layout = GridspecLayout(n_rows=4, n_columns=2, height='200px')
>>> layout[:3, 0] = Button(layout=Layout(height='auto', width='auto'))
>>> layout[1:, 1] = Button(layout=Layout(height='auto', width='auto'))
>>> layout[-1, 0] = Button(layout=Layout(height='auto', width='auto'))
>>> layout[0, 1] = Button(layout=Layout(height='auto', width='auto'))
>>> layout

## Class: TwoByTwoLayout

**Description:** Define a layout with 2x2 regular grid.

Parameters
----------

top_left: instance of Widget
top_right: instance of Widget
bottom_left: instance of Widget
bottom_right: instance of Widget
    widgets to fill the positions in the layout

merge: bool
    flag to say whether the empty positions should be automatically merged

{style_params}

Examples
--------

>>> from ipywidgets import TwoByTwoLayout, Button
>>> TwoByTwoLayout(top_left=Button(description="Top left"),
...                top_right=Button(description="Top right"),
...                bottom_left=Button(description="Bottom left"),
...                bottom_right=Button(description="Bottom right"))

### Function: __init__(self)

### Function: _delegate_to_layout(self, change)

**Description:** delegate the trait types to their counterparts in self.layout

### Function: _set_observers(self)

**Description:** set observers on all layout properties defined in this class

### Function: _copy_layout_props(self)

### Function: __init__(self)

### Function: _size_to_css(size)

### Function: _convert_sizes(self, size_list)

### Function: _update_layout(self)

### Function: _child_changed(self, change)

### Function: __init__(self, n_rows, n_columns)

### Function: _validate_integer(self, proposal)

### Function: _get_indices_from_slice(self, row, column)

**Description:** convert a two-dimensional slice to a list of rows and column indices

### Function: __setitem__(self, key, value)

### Function: __getitem__(self, key)

### Function: _update_layout(self)

### Function: __init__(self)

### Function: _update_layout(self)

### Function: _child_changed(self, change)
