## AI Summary

A file named _tricontour.py.


## Class: TriContourSet

**Description:** Create and store a set of contour lines or filled regions for
a triangular grid.

This class is typically not instantiated directly by the user but by
`~.Axes.tricontour` and `~.Axes.tricontourf`.

%(contour_set_attributes)s

### Function: tricontour(ax)

**Description:** %(_tricontour_doc)s

linewidths : float or array-like, default: :rc:`contour.linewidth`
    The line width of the contour lines.

    If a number, all levels will be plotted with this linewidth.

    If a sequence, the levels in ascending order will be plotted with
    the linewidths in the order specified.

    If None, this falls back to :rc:`lines.linewidth`.

linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
    If *linestyles* is *None*, the default is 'solid' unless the lines are
    monochrome.  In that case, negative contours will take their linestyle
    from :rc:`contour.negative_linestyle` setting.

    *linestyles* can also be an iterable of the above strings specifying a
    set of linestyles to be used. If this iterable is shorter than the
    number of contour levels it will be repeated as necessary.

### Function: tricontourf(ax)

**Description:** %(_tricontour_doc)s

hatches : list[str], optional
    A list of crosshatch patterns to use on the filled areas.
    If None, no hatching will be added to the contour.

Notes
-----
`.tricontourf` fills intervals that are closed at the top; that is, for
boundaries *z1* and *z2*, the filled region is::

    z1 < Z <= z2

except for the lowest interval, which is closed on both sides (i.e. it
includes the lowest value).

### Function: __init__(self, ax)

**Description:** Draw triangular grid contour lines or filled regions,
depending on whether keyword arg *filled* is False
(default) or True.

The first argument of the initializer must be an `~.axes.Axes`
object.  The remaining arguments and keyword arguments
are described in the docstring of `~.Axes.tricontour`.

### Function: _process_args(self)

**Description:** Process args and kwargs.

### Function: _contour_args(self, args, kwargs)
