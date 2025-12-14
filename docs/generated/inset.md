## AI Summary

A file named inset.py.


## Class: InsetIndicator

**Description:** An artist to highlight an area of interest.

An inset indicator is a rectangle on the plot at the position indicated by
*bounds* that optionally has lines that connect the rectangle to an inset
Axes (`.Axes.inset_axes`).

.. versionadded:: 3.10

### Function: __init__(self, bounds, inset_ax, zorder)

**Description:** Parameters
----------
bounds : [x0, y0, width, height], optional
    Lower-left corner of rectangle to be marked, and its width
    and height.  If not set, the bounds will be calculated from the
    data limits of inset_ax, which must be supplied.

inset_ax : `~.axes.Axes`, optional
    An optional inset Axes to draw connecting lines to.  Two lines are
    drawn connecting the indicator box to the inset Axes on corners
    chosen so as to not overlap with the indicator box.

zorder : float, default: 4.99
    Drawing order of the rectangle and connector lines.  The default,
    4.99, is just below the default level of inset Axes.

**kwargs
    Other keyword arguments are passed on to the `.Rectangle` patch.

### Function: _shared_setter(self, prop, val)

**Description:** Helper function to set the same style property on the artist and its children.

### Function: set_alpha(self, alpha)

### Function: set_edgecolor(self, color)

**Description:** Set the edge color of the rectangle and the connectors.

Parameters
----------
color : :mpltype:`color` or None

### Function: set_color(self, c)

**Description:** Set the edgecolor of the rectangle and the connectors, and the
facecolor for the rectangle.

Parameters
----------
c : :mpltype:`color`

### Function: set_linewidth(self, w)

**Description:** Set the linewidth in points of the rectangle and the connectors.

Parameters
----------
w : float or None

### Function: set_linestyle(self, ls)

**Description:** Set the linestyle of the rectangle and the connectors.

==========================================  =================
linestyle                                   description
==========================================  =================
``'-'`` or ``'solid'``                      solid line
``'--'`` or ``'dashed'``                    dashed line
``'-.'`` or ``'dashdot'``                   dash-dotted line
``':'`` or ``'dotted'``                     dotted line
``'none'``, ``'None'``, ``' '``, or ``''``  draw nothing
==========================================  =================

Alternatively a dash tuple of the following form can be provided::

    (offset, onoffseq)

where ``onoffseq`` is an even length tuple of on and off ink in points.

Parameters
----------
ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
    The line style.

### Function: _bounds_from_inset_ax(self)

### Function: _update_connectors(self)

### Function: rectangle(self)

**Description:** `.Rectangle`: the indicator frame.

### Function: connectors(self)

**Description:** 4-tuple of `.patches.ConnectionPatch` or None
    The four connector lines connecting to (lower_left, upper_left,
    lower_right upper_right) corners of *inset_ax*. Two lines are
    set with visibility to *False*,  but the user can set the
    visibility to True if the automatic choice is not deemed correct.

### Function: draw(self, renderer)

### Function: __getitem__(self, key)
