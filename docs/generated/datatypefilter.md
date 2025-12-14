## AI Summary

A file named datatypefilter.py.


## Class: DataTypeFilter

**Description:** Returns the preferred display format

### Function: __call__(self, output)

**Description:** Return the first available format in the priority.

Produces a UserWarning if no compatible mimetype is found.

`output` is dict with structure {mimetype-of-element: value-of-element}
