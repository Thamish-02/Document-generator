## AI Summary

A file named lines.py.


### Function: _get_dash_pattern(style)

**Description:** Convert linestyle to dash pattern.

### Function: _get_dash_patterns(styles)

**Description:** Convert linestyle or sequence of linestyles to list of dash patterns.

### Function: _get_inverse_dash_pattern(offset, dashes)

**Description:** Return the inverse of the given dash pattern, for filling the gaps.

### Function: _scale_dashes(offset, dashes, lw)

### Function: segment_hits(cx, cy, x, y, radius)

**Description:** Return the indices of the segments in the polyline with coordinates (*cx*,
*cy*) that are within a distance *radius* of the point (*x*, *y*).

### Function: _mark_every_path(markevery, tpath, affine, ax)

**Description:** Helper function that sorts out how to deal the input
`markevery` and returns the points where markers should be drawn.

Takes in the `markevery` value and the line path and returns the
sub-sampled path.

## Class: Line2D

**Description:** A line - the line can have both a solid linestyle connecting all
the vertices, and a marker at each vertex.  Additionally, the
drawing of the solid line is influenced by the drawstyle, e.g., one
can create "stepped" lines in various styles.

## Class: AxLine

**Description:** A helper class that implements `~.Axes.axline`, by recomputing the artist
transform at draw time.

## Class: VertexSelector

**Description:** Manage the callbacks to maintain a list of selected vertices for `.Line2D`.
Derived classes should override the `process_selected` method to do
something with the picks.

Here is an example which highlights the selected verts with red circles::

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as lines

    class HighlightSelected(lines.VertexSelector):
        def __init__(self, line, fmt='ro', **kwargs):
            super().__init__(line)
            self.markers, = self.axes.plot([], [], fmt, **kwargs)

        def process_selected(self, ind, xs, ys):
            self.markers.set_data(xs, ys)
            self.canvas.draw()

    fig, ax = plt.subplots()
    x, y = np.random.rand(2, 30)
    line, = ax.plot(x, y, 'bs-', picker=5)

    selector = HighlightSelected(line)
    plt.show()

### Function: _slice_or_none(in_v, slc)

**Description:** Helper function to cope with `codes` being an ndarray or `None`.

### Function: __str__(self)

### Function: __init__(self, xdata, ydata)

**Description:** Create a `.Line2D` instance with *x* and *y* data in sequences of
*xdata*, *ydata*.

Additional keyword arguments are `.Line2D` properties:

%(Line2D:kwdoc)s

See :meth:`set_linestyle` for a description of the line styles,
:meth:`set_marker` for a description of the markers, and
:meth:`set_drawstyle` for a description of the draw styles.

### Function: contains(self, mouseevent)

**Description:** Test whether *mouseevent* occurred on the line.

An event is deemed to have occurred "on" the line if it is less
than ``self.pickradius`` (default: 5 points) away from it.  Use
`~.Line2D.get_pickradius` or `~.Line2D.set_pickradius` to get or set
the pick radius.

Parameters
----------
mouseevent : `~matplotlib.backend_bases.MouseEvent`

Returns
-------
contains : bool
    Whether any values are within the radius.
details : dict
    A dictionary ``{'ind': pointlist}``, where *pointlist* is a
    list of points of the line that are within the pickradius around
    the event position.

    TODO: sort returned indices by distance

### Function: get_pickradius(self)

**Description:** Return the pick radius used for containment tests.

See `.contains` for more details.

### Function: set_pickradius(self, pickradius)

**Description:** Set the pick radius used for containment tests.

See `.contains` for more details.

Parameters
----------
pickradius : float
    Pick radius, in points.

### Function: get_fillstyle(self)

**Description:** Return the marker fill style.

See also `~.Line2D.set_fillstyle`.

### Function: set_fillstyle(self, fs)

**Description:** Set the marker fill style.

Parameters
----------
fs : {'full', 'left', 'right', 'bottom', 'top', 'none'}
    Possible values:

    - 'full': Fill the whole marker with the *markerfacecolor*.
    - 'left', 'right', 'bottom', 'top': Fill the marker half at
      the given side with the *markerfacecolor*. The other
      half of the marker is filled with *markerfacecoloralt*.
    - 'none': No filling.

    For examples see :ref:`marker_fill_styles`.

### Function: set_markevery(self, every)

**Description:** Set the markevery property to subsample the plot when using markers.

e.g., if ``every=5``, every 5-th marker will be plotted.

Parameters
----------
every : None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]
    Which markers to plot.

    - ``every=None``: every point will be plotted.
    - ``every=N``: every N-th marker will be plotted starting with
      marker 0.
    - ``every=(start, N)``: every N-th marker, starting at index
      *start*, will be plotted.
    - ``every=slice(start, end, N)``: every N-th marker, starting at
      index *start*, up to but not including index *end*, will be
      plotted.
    - ``every=[i, j, m, ...]``: only markers at the given indices
      will be plotted.
    - ``every=[True, False, True, ...]``: only positions that are True
      will be plotted. The list must have the same length as the data
      points.
    - ``every=0.1``, (i.e. a float): markers will be spaced at
      approximately equal visual distances along the line; the distance
      along the line between markers is determined by multiplying the
      display-coordinate distance of the Axes bounding-box diagonal
      by the value of *every*.
    - ``every=(0.5, 0.1)`` (i.e. a length-2 tuple of float): similar
      to ``every=0.1`` but the first marker will be offset along the
      line by 0.5 multiplied by the
      display-coordinate-diagonal-distance along the line.

    For examples see
    :doc:`/gallery/lines_bars_and_markers/markevery_demo`.

Notes
-----
Setting *markevery* will still only draw markers at actual data points.
While the float argument form aims for uniform visual spacing, it has
to coerce from the ideal spacing to the nearest available data point.
Depending on the number and distribution of data points, the result
may still not look evenly spaced.

When using a start offset to specify the first marker, the offset will
be from the first data point which may be different from the first
the visible data point if the plot is zoomed in.

If zooming in on a plot when using float arguments then the actual
data points that have markers will change because the distance between
markers is always determined from the display-coordinates
axes-bounding-box-diagonal regardless of the actual axes data limits.

### Function: get_markevery(self)

**Description:** Return the markevery setting for marker subsampling.

See also `~.Line2D.set_markevery`.

### Function: set_picker(self, p)

**Description:** Set the event picker details for the line.

Parameters
----------
p : float or callable[[Artist, Event], tuple[bool, dict]]
    If a float, it is used as the pick radius in points.

### Function: get_bbox(self)

**Description:** Get the bounding box of this line.

### Function: get_window_extent(self, renderer)

### Function: set_data(self)

**Description:** Set the x and y data.

Parameters
----------
*args : (2, N) array or two 1D arrays

See Also
--------
set_xdata
set_ydata

### Function: recache_always(self)

### Function: recache(self, always)

### Function: _transform_path(self, subslice)

**Description:** Put a TransformedPath instance at self._transformed_path;
all invalidation of the transform is then handled by the
TransformedPath instance.

### Function: _get_transformed_path(self)

**Description:** Return this line's `~matplotlib.transforms.TransformedPath`.

### Function: set_transform(self, t)

### Function: draw(self, renderer)

### Function: get_antialiased(self)

**Description:** Return whether antialiased rendering is used.

### Function: get_color(self)

**Description:** Return the line color.

See also `~.Line2D.set_color`.

### Function: get_drawstyle(self)

**Description:** Return the drawstyle.

See also `~.Line2D.set_drawstyle`.

### Function: get_gapcolor(self)

**Description:** Return the line gapcolor.

See also `~.Line2D.set_gapcolor`.

### Function: get_linestyle(self)

**Description:** Return the linestyle.

See also `~.Line2D.set_linestyle`.

### Function: get_linewidth(self)

**Description:** Return the linewidth in points.

See also `~.Line2D.set_linewidth`.

### Function: get_marker(self)

**Description:** Return the line marker.

See also `~.Line2D.set_marker`.

### Function: get_markeredgecolor(self)

**Description:** Return the marker edge color.

See also `~.Line2D.set_markeredgecolor`.

### Function: get_markeredgewidth(self)

**Description:** Return the marker edge width in points.

See also `~.Line2D.set_markeredgewidth`.

### Function: _get_markerfacecolor(self, alt)

### Function: get_markerfacecolor(self)

**Description:** Return the marker face color.

See also `~.Line2D.set_markerfacecolor`.

### Function: get_markerfacecoloralt(self)

**Description:** Return the alternate marker face color.

See also `~.Line2D.set_markerfacecoloralt`.

### Function: get_markersize(self)

**Description:** Return the marker size in points.

See also `~.Line2D.set_markersize`.

### Function: get_data(self, orig)

**Description:** Return the line data as an ``(xdata, ydata)`` pair.

If *orig* is *True*, return the original data.

### Function: get_xdata(self, orig)

**Description:** Return the xdata.

If *orig* is *True*, return the original data, else the
processed data.

### Function: get_ydata(self, orig)

**Description:** Return the ydata.

If *orig* is *True*, return the original data, else the
processed data.

### Function: get_path(self)

**Description:** Return the `~matplotlib.path.Path` associated with this line.

### Function: get_xydata(self)

**Description:** Return the *xy* data as a (N, 2) array.

### Function: set_antialiased(self, b)

**Description:** Set whether to use antialiased rendering.

Parameters
----------
b : bool

### Function: set_color(self, color)

**Description:** Set the color of the line.

Parameters
----------
color : :mpltype:`color`

### Function: set_drawstyle(self, drawstyle)

**Description:** Set the drawstyle of the plot.

The drawstyle determines how the points are connected.

Parameters
----------
drawstyle : {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}, default: 'default'
    For 'default', the points are connected with straight lines.

    The steps variants connect the points with step-like lines,
    i.e. horizontal lines with vertical steps. They differ in the
    location of the step:

    - 'steps-pre': The step is at the beginning of the line segment,
      i.e. the line will be at the y-value of point to the right.
    - 'steps-mid': The step is halfway between the points.
    - 'steps-post: The step is at the end of the line segment,
      i.e. the line will be at the y-value of the point to the left.
    - 'steps' is equal to 'steps-pre' and is maintained for
      backward-compatibility.

    For examples see :doc:`/gallery/lines_bars_and_markers/step_demo`.

### Function: set_gapcolor(self, gapcolor)

**Description:** Set a color to fill the gaps in the dashed line style.

.. note::

    Striped lines are created by drawing two interleaved dashed lines.
    There can be overlaps between those two, which may result in
    artifacts when using transparency.

    This functionality is experimental and may change.

Parameters
----------
gapcolor : :mpltype:`color` or None
    The color with which to fill the gaps. If None, the gaps are
    unfilled.

### Function: set_linewidth(self, w)

**Description:** Set the line width in points.

Parameters
----------
w : float
    Line width, in points.

### Function: set_linestyle(self, ls)

**Description:** Set the linestyle of the line.

Parameters
----------
ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
    Possible values:

    - A string:

      ==========================================  =================
      linestyle                                   description
      ==========================================  =================
      ``'-'`` or ``'solid'``                      solid line
      ``'--'`` or  ``'dashed'``                   dashed line
      ``'-.'`` or  ``'dashdot'``                  dash-dotted line
      ``':'`` or ``'dotted'``                     dotted line
      ``'none'``, ``'None'``, ``' '``, or ``''``  draw nothing
      ==========================================  =================

    - Alternatively a dash tuple of the following form can be
      provided::

          (offset, onoffseq)

      where ``onoffseq`` is an even length tuple of on and off ink
      in points. See also :meth:`set_dashes`.

    For examples see :doc:`/gallery/lines_bars_and_markers/linestyles`.

### Function: set_marker(self, marker)

**Description:** Set the line marker.

Parameters
----------
marker : marker style string, `~.path.Path` or `~.markers.MarkerStyle`
    See `~matplotlib.markers` for full description of possible
    arguments.

### Function: _set_markercolor(self, name, has_rcdefault, val)

### Function: set_markeredgecolor(self, ec)

**Description:** Set the marker edge color.

Parameters
----------
ec : :mpltype:`color`

### Function: set_markerfacecolor(self, fc)

**Description:** Set the marker face color.

Parameters
----------
fc : :mpltype:`color`

### Function: set_markerfacecoloralt(self, fc)

**Description:** Set the alternate marker face color.

Parameters
----------
fc : :mpltype:`color`

### Function: set_markeredgewidth(self, ew)

**Description:** Set the marker edge width in points.

Parameters
----------
ew : float
     Marker edge width, in points.

### Function: set_markersize(self, sz)

**Description:** Set the marker size in points.

Parameters
----------
sz : float
     Marker size, in points.

### Function: set_xdata(self, x)

**Description:** Set the data array for x.

Parameters
----------
x : 1D array

See Also
--------
set_data
set_ydata

### Function: set_ydata(self, y)

**Description:** Set the data array for y.

Parameters
----------
y : 1D array

See Also
--------
set_data
set_xdata

### Function: set_dashes(self, seq)

**Description:** Set the dash sequence.

The dash sequence is a sequence of floats of even length describing
the length of dashes and spaces in points.

For example, (5, 2, 1, 2) describes a sequence of 5 point and 1 point
dashes separated by 2 point spaces.

See also `~.Line2D.set_gapcolor`, which allows those spaces to be
filled with a color.

Parameters
----------
seq : sequence of floats (on/off ink in points) or (None, None)
    If *seq* is empty or ``(None, None)``, the linestyle will be set
    to solid.

### Function: update_from(self, other)

**Description:** Copy properties from *other* to self.

### Function: set_dash_joinstyle(self, s)

**Description:** How to join segments of the line if it `~Line2D.is_dashed`.

The default joinstyle is :rc:`lines.dash_joinstyle`.

Parameters
----------
s : `.JoinStyle` or %(JoinStyle)s

### Function: set_solid_joinstyle(self, s)

**Description:** How to join segments if the line is solid (not `~Line2D.is_dashed`).

The default joinstyle is :rc:`lines.solid_joinstyle`.

Parameters
----------
s : `.JoinStyle` or %(JoinStyle)s

### Function: get_dash_joinstyle(self)

**Description:** Return the `.JoinStyle` for dashed lines.

See also `~.Line2D.set_dash_joinstyle`.

### Function: get_solid_joinstyle(self)

**Description:** Return the `.JoinStyle` for solid lines.

See also `~.Line2D.set_solid_joinstyle`.

### Function: set_dash_capstyle(self, s)

**Description:** How to draw the end caps if the line is `~Line2D.is_dashed`.

The default capstyle is :rc:`lines.dash_capstyle`.

Parameters
----------
s : `.CapStyle` or %(CapStyle)s

### Function: set_solid_capstyle(self, s)

**Description:** How to draw the end caps if the line is solid (not `~Line2D.is_dashed`)

The default capstyle is :rc:`lines.solid_capstyle`.

Parameters
----------
s : `.CapStyle` or %(CapStyle)s

### Function: get_dash_capstyle(self)

**Description:** Return the `.CapStyle` for dashed lines.

See also `~.Line2D.set_dash_capstyle`.

### Function: get_solid_capstyle(self)

**Description:** Return the `.CapStyle` for solid lines.

See also `~.Line2D.set_solid_capstyle`.

### Function: is_dashed(self)

**Description:** Return whether line has a dashed linestyle.

A custom linestyle is assumed to be dashed, we do not inspect the
``onoffseq`` directly.

See also `~.Line2D.set_linestyle`.

### Function: __init__(self, xy1, xy2, slope)

**Description:** Parameters
----------
xy1 : (float, float)
    The first set of (x, y) coordinates for the line to pass through.
xy2 : (float, float) or None
    The second set of (x, y) coordinates for the line to pass through.
    Both *xy2* and *slope* must be passed, but one of them must be None.
slope : float or None
    The slope of the line. Both *xy2* and *slope* must be passed, but one of
    them must be None.

### Function: get_transform(self)

### Function: draw(self, renderer)

### Function: get_xy1(self)

**Description:** Return the *xy1* value of the line.

### Function: get_xy2(self)

**Description:** Return the *xy2* value of the line.

### Function: get_slope(self)

**Description:** Return the *slope* value of the line.

### Function: set_xy1(self)

**Description:** Set the *xy1* value of the line.

Parameters
----------
xy1 : tuple[float, float]
    Points for the line to pass through.

### Function: set_xy2(self)

**Description:** Set the *xy2* value of the line.

.. note::

    You can only set *xy2* if the line was created using the *xy2*
    parameter. If the line was created using *slope*, please use
    `~.AxLine.set_slope`.

Parameters
----------
xy2 : tuple[float, float]
    Points for the line to pass through.

### Function: set_slope(self, slope)

**Description:** Set the *slope* value of the line.

.. note::

    You can only set *slope* if the line was created using the *slope*
    parameter. If the line was created using *xy2*, please use
    `~.AxLine.set_xy2`.

Parameters
----------
slope : float
    The slope of the line.

### Function: __init__(self, line)

**Description:** Parameters
----------
line : `~matplotlib.lines.Line2D`
    The line must already have been added to an `~.axes.Axes` and must
    have its picker property set.

### Function: process_selected(self, ind, xs, ys)

**Description:** Default "do nothing" implementation of the `process_selected` method.

Parameters
----------
ind : list of int
    The indices of the selected vertices.
xs, ys : array-like
    The coordinates of the selected vertices.

### Function: onpick(self, event)

**Description:** When the line is picked, update the set of selected indices.
