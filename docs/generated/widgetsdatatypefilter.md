## AI Summary

A file named widgetsdatatypefilter.py.


## Class: WidgetsDataTypeFilter

**Description:** Returns the preferred display format, excluding the widget output if
there is no widget state available

### Function: __init__(self, notebook_metadata, resources)

**Description:** Initialize the filter.

### Function: __call__(self, output)

**Description:** Return the first available format in the priority.

Produces a UserWarning if no compatible mimetype is found.

`output` is dict with structure {mimetype-of-element: value-of-element}
