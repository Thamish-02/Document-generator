## AI Summary

A file named widget_box.py.


## Class: Box

**Description:** Displays multiple widgets in a group.

The widgets are laid out horizontally.

Parameters
----------
{box_params}

Examples
--------
>>> import ipywidgets as widgets
>>> title_widget = widgets.HTML('<em>Box Example</em>')
>>> slider = widgets.IntSlider()
>>> widgets.Box([title_widget, slider])

## Class: VBox

**Description:** Displays multiple widgets vertically using the flexible box model.

Parameters
----------
{box_params}

Examples
--------
>>> import ipywidgets as widgets
>>> title_widget = widgets.HTML('<em>Vertical Box Example</em>')
>>> slider = widgets.IntSlider()
>>> widgets.VBox([title_widget, slider])

## Class: HBox

**Description:** Displays multiple widgets horizontally using the flexible box model.

Parameters
----------
{box_params}

Examples
--------
>>> import ipywidgets as widgets
>>> title_widget = widgets.HTML('<em>Horizontal Box Example</em>')
>>> slider = widgets.IntSlider()
>>> widgets.HBox([title_widget, slider])

## Class: GridBox

**Description:** Displays multiple widgets in rows and columns using the grid box model.

Parameters
----------
{box_params}

Examples
--------
>>> import ipywidgets as widgets
>>> title_widget = widgets.HTML('<em>Grid Box Example</em>')
>>> slider = widgets.IntSlider()
>>> button1 = widgets.Button(description='1')
>>> button2 = widgets.Button(description='2')
>>> # Create a grid with two columns, splitting space equally
>>> layout = widgets.Layout(grid_template_columns='1fr 1fr')
>>> widgets.GridBox([title_widget, slider, button1, button2], layout=layout)

### Function: __init__(self, children)
