## AI Summary

A file named _secondary_axes.py.


## Class: SecondaryAxis

**Description:** General class to hold a Secondary_X/Yaxis.

### Function: __init__(self, parent, orientation, location, functions, transform)

**Description:** See `.secondary_xaxis` and `.secondary_yaxis` for the doc string.
While there is no need for this to be private, it should really be
called by those higher level functions.

### Function: set_alignment(self, align)

**Description:** Set if axes spine and labels are drawn at top or bottom (or left/right)
of the Axes.

Parameters
----------
align : {'top', 'bottom', 'left', 'right'}
    Either 'top' or 'bottom' for orientation='x' or
    'left' or 'right' for orientation='y' axis.

### Function: set_location(self, location, transform)

**Description:** Set the vertical or horizontal location of the axes in
parent-normalized coordinates.

Parameters
----------
location : {'top', 'bottom', 'left', 'right'} or float
    The position to put the secondary axis.  Strings can be 'top' or
    'bottom' for orientation='x' and 'right' or 'left' for
    orientation='y'. A float indicates the relative position on the
    parent Axes to put the new Axes, 0.0 being the bottom (or left)
    and 1.0 being the top (or right).

transform : `.Transform`, optional
    Transform for the location to use. Defaults to
    the parent's ``transAxes``, so locations are normally relative to
    the parent axes.

    .. versionadded:: 3.9

### Function: apply_aspect(self, position)

### Function: set_ticks(self, ticks, labels)

### Function: set_functions(self, functions)

**Description:** Set how the secondary axis converts limits from the parent Axes.

Parameters
----------
functions : 2-tuple of func, or `Transform` with an inverse.
    Transform between the parent axis values and the secondary axis
    values.

    If supplied as a 2-tuple of functions, the first function is
    the forward transform function and the second is the inverse
    transform.

    If a transform is supplied, then the transform must have an
    inverse.

### Function: draw(self, renderer)

**Description:** Draw the secondary Axes.

Consults the parent Axes for its limits and converts them
using the converter specified by
`~.axes._secondary_axes.set_functions` (or *functions*
parameter when Axes initialized.)

### Function: _set_scale(self)

**Description:** Check if parent has set its scale

### Function: _set_lims(self)

**Description:** Set the limits based on parent limits and the convert method
between the parent and this secondary Axes.

### Function: set_aspect(self)

**Description:** Secondary Axes cannot set the aspect ratio, so calling this just
sets a warning.

### Function: set_color(self, color)

**Description:** Change the color of the secondary Axes and all decorators.

Parameters
----------
color : :mpltype:`color`
