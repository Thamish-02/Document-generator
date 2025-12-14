## AI Summary

A file named contour.py.


### Function: _contour_labeler_event_handler(cs, inline, inline_spacing, event)

## Class: ContourLabeler

**Description:** Mixin to provide labelling capability to `.ContourSet`.

### Function: _find_closest_point_on_path(xys, p)

**Description:** Parameters
----------
xys : (N, 2) array-like
    Coordinates of vertices.
p : (float, float)
    Coordinates of point.

Returns
-------
d2min : float
    Minimum square distance of *p* to *xys*.
proj : (float, float)
    Projection of *p* onto *xys*.
imin : (int, int)
    Consecutive indices of vertices of segment in *xys* where *proj* is.
    Segments are considered as including their end-points; i.e. if the
    closest point on the path is a node in *xys* with index *i*, this
    returns ``(i-1, i)``.  For the special case where *xys* is a single
    point, this returns ``(0, 0)``.

## Class: ContourSet

**Description:** Store a set of contour lines or filled regions.

User-callable method: `~.Axes.clabel`

Parameters
----------
ax : `~matplotlib.axes.Axes`

levels : [level0, level1, ..., leveln]
    A list of floating point numbers indicating the contour levels.

allsegs : [level0segs, level1segs, ...]
    List of all the polygon segments for all the *levels*.
    For contour lines ``len(allsegs) == len(levels)``, and for
    filled contour regions ``len(allsegs) = len(levels)-1``. The lists
    should look like ::

        level0segs = [polygon0, polygon1, ...]
        polygon0 = [[x0, y0], [x1, y1], ...]

allkinds : ``None`` or [level0kinds, level1kinds, ...]
    Optional list of all the polygon vertex kinds (code types), as
    described and used in Path. This is used to allow multiply-
    connected paths such as holes within filled polygons.
    If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
    should look like ::

        level0kinds = [polygon0kinds, ...]
        polygon0kinds = [vertexcode0, vertexcode1, ...]

    If *allkinds* is not ``None``, usually all polygons for a
    particular contour level are grouped together so that
    ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

**kwargs
    Keyword arguments are as described in the docstring of
    `~.Axes.contour`.

%(contour_set_attributes)s

## Class: QuadContourSet

**Description:** Create and store a set of contour lines or filled regions.

This class is typically not instantiated directly by the user but by
`~.Axes.contour` and `~.Axes.contourf`.

%(contour_set_attributes)s

### Function: clabel(self, levels)

**Description:** Label a contour plot.

Adds labels to line contours in this `.ContourSet` (which inherits from
this mixin class).

Parameters
----------
levels : array-like, optional
    A list of level values, that should be labeled. The list must be
    a subset of ``cs.levels``. If not given, all levels are labeled.

fontsize : str or float, default: :rc:`font.size`
    Size in points or relative size e.g., 'smaller', 'x-large'.
    See `.Text.set_size` for accepted string values.

colors : :mpltype:`color` or colors or None, default: None
    The label colors:

    - If *None*, the color of each label matches the color of
      the corresponding contour.

    - If one string color, e.g., *colors* = 'r' or *colors* =
      'red', all labels will be plotted in this color.

    - If a tuple of colors (string, float, RGB, etc), different labels
      will be plotted in different colors in the order specified.

inline : bool, default: True
    If ``True`` the underlying contour is removed where the label is
    placed.

inline_spacing : float, default: 5
    Space in pixels to leave on each side of label when placing inline.

    This spacing will be exact for labels at locations where the
    contour is straight, less so for labels on curved contours.

fmt : `.Formatter` or str or callable or dict, optional
    How the levels are formatted:

    - If a `.Formatter`, it is used to format all levels at once, using
      its `.Formatter.format_ticks` method.
    - If a str, it is interpreted as a %-style format string.
    - If a callable, it is called with one level at a time and should
      return the corresponding label.
    - If a dict, it should directly map levels to labels.

    The default is to use a standard `.ScalarFormatter`.

manual : bool or iterable, default: False
    If ``True``, contour labels will be placed manually using
    mouse clicks. Click the first button near a contour to
    add a label, click the second button (or potentially both
    mouse buttons at once) to finish adding labels. The third
    button can be used to remove the last label added, but
    only if labels are not inline. Alternatively, the keyboard
    can be used to select label locations (enter to end label
    placement, delete or backspace act like the third mouse button,
    and any other key will select a label location).

    *manual* can also be an iterable object of (x, y) tuples.
    Contour labels will be created as if mouse is clicked at each
    (x, y) position.

rightside_up : bool, default: True
    If ``True``, label rotations will always be plus
    or minus 90 degrees from level.

use_clabeltext : bool, default: False
    If ``True``, use `.Text.set_transform_rotates_text` to ensure that
    label rotation is updated whenever the Axes aspect changes.

zorder : float or None, default: ``(2 + contour.get_zorder())``
    zorder of the contour labels.

Returns
-------
labels
    A list of `.Text` instances for the labels.

### Function: print_label(self, linecontour, labelwidth)

**Description:** Return whether a contour is long enough to hold a label.

### Function: too_close(self, x, y, lw)

**Description:** Return whether a label is already near this location.

### Function: _get_nth_label_width(self, nth)

**Description:** Return the width of the *nth* label, in pixels.

### Function: get_text(self, lev, fmt)

**Description:** Get the text of the label.

### Function: locate_label(self, linecontour, labelwidth)

**Description:** Find good place to draw a label (relatively flat part of the contour).

### Function: _split_path_and_get_label_rotation(self, path, idx, screen_pos, lw, spacing)

**Description:** Prepare for insertion of a label at index *idx* of *path*.

Parameters
----------
path : Path
    The path where the label will be inserted, in data space.
idx : int
    The vertex index after which the label will be inserted.
screen_pos : (float, float)
    The position where the label will be inserted, in screen space.
lw : float
    The label width, in screen space.
spacing : float
    Extra spacing around the label, in screen space.

Returns
-------
path : Path
    The path, broken so that the label can be drawn over it.
angle : float
    The rotation of the label.

Notes
-----
Both tasks are done together to avoid calculating path lengths multiple times,
which is relatively costly.

The method used here involves computing the path length along the contour in
pixel coordinates and then looking (label width / 2) away from central point to
determine rotation and then to break contour if desired.  The extra spacing is
taken into account when breaking the path, but not when computing the angle.

### Function: add_label(self, x, y, rotation, lev, cvalue)

**Description:** Add a contour label, respecting whether *use_clabeltext* was set.

### Function: add_label_near(self, x, y, inline, inline_spacing, transform)

**Description:** Add a label near the point ``(x, y)``.

Parameters
----------
x, y : float
    The approximate location of the label.
inline : bool, default: True
    If *True* remove the segment of the contour beneath the label.
inline_spacing : int, default: 5
    Space in pixels to leave on each side of label when placing
    inline. This spacing will be exact for labels at locations where
    the contour is straight, less so for labels on curved contours.
transform : `.Transform` or `False`, default: ``self.axes.transData``
    A transform applied to ``(x, y)`` before labeling.  The default
    causes ``(x, y)`` to be interpreted as data coordinates.  `False`
    is a synonym for `.IdentityTransform`; i.e. ``(x, y)`` should be
    interpreted as display coordinates.

### Function: pop_label(self, index)

**Description:** Defaults to removing last label, but any index can be supplied

### Function: labels(self, inline, inline_spacing)

### Function: remove(self)

### Function: __init__(self, ax)

**Description:** Draw contour lines or filled regions, depending on
whether keyword arg *filled* is ``False`` (default) or ``True``.

Call signature::

    ContourSet(ax, levels, allsegs, [allkinds], **kwargs)

Parameters
----------
ax : `~matplotlib.axes.Axes`
    The `~.axes.Axes` object to draw on.

levels : [level0, level1, ..., leveln]
    A list of floating point numbers indicating the contour
    levels.

allsegs : [level0segs, level1segs, ...]
    List of all the polygon segments for all the *levels*.
    For contour lines ``len(allsegs) == len(levels)``, and for
    filled contour regions ``len(allsegs) = len(levels)-1``. The lists
    should look like ::

        level0segs = [polygon0, polygon1, ...]
        polygon0 = [[x0, y0], [x1, y1], ...]

allkinds : [level0kinds, level1kinds, ...], optional
    Optional list of all the polygon vertex kinds (code types), as
    described and used in Path. This is used to allow multiply-
    connected paths such as holes within filled polygons.
    If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
    should look like ::

        level0kinds = [polygon0kinds, ...]
        polygon0kinds = [vertexcode0, vertexcode1, ...]

    If *allkinds* is not ``None``, usually all polygons for a
    particular contour level are grouped together so that
    ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

**kwargs
    Keyword arguments are as described in the docstring of
    `~.Axes.contour`.

### Function: get_transform(self)

**Description:** Return the `.Transform` instance used by this ContourSet.

### Function: __getstate__(self)

### Function: legend_elements(self, variable_name, str_format)

**Description:** Return a list of artists and labels suitable for passing through
to `~.Axes.legend` which represent this ContourSet.

The labels have the form "0 < x <= 1" stating the data ranges which
the artists represent.

Parameters
----------
variable_name : str
    The string used inside the inequality used on the labels.
str_format : function: float -> str
    Function used to format the numbers in the labels.

Returns
-------
artists : list[`.Artist`]
    A list of the artists.
labels : list[str]
    A list of the labels.

### Function: _process_args(self)

**Description:** Process *args* and *kwargs*; override in derived classes.

Must set self.levels, self.zmin and self.zmax, and update Axes limits.

### Function: _make_paths_from_contour_generator(self)

**Description:** Compute ``paths`` using C extension.

### Function: _get_lowers_and_uppers(self)

**Description:** Return ``(lowers, uppers)`` for filled contours.

### Function: changed(self)

### Function: _autolev(self, N)

**Description:** Select contour levels to span the data.

The target number of levels, *N*, is used only when the
scale is not log and default locator is used.

We need two more levels for filled contours than for
line contours, because for the latter we need to specify
the lower and upper boundary of each range. For example,
a single contour boundary, say at z = 0, requires only
one contour line, but two filled regions, and therefore
three levels to provide boundaries for both regions.

### Function: _process_contour_level_args(self, args, z_dtype)

**Description:** Determine the contour levels and store in self.levels.

### Function: _process_levels(self)

**Description:** Assign values to :attr:`layers` based on :attr:`levels`,
adding extended layers as needed if contours are filled.

For line contours, layers simply coincide with levels;
a line is a thin layer.  No extended levels are needed
with line contours.

### Function: _process_colors(self)

**Description:** Color argument processing for contouring.

Note that we base the colormapping on the contour levels
and layers, not on the actual range of the Z values.  This
means we don't have to worry about bad values in Z, and we
always have the full dynamic range available for the selected
levels.

The color is based on the midpoint of the layer, except for
extended end layers.  By default, the norm vmin and vmax
are the extreme values of the non-extended levels.  Hence,
the layer color extremes are not the extreme values of
the colormap itself, but approach those values as the number
of levels increases.  An advantage of this scheme is that
line contours, when added to filled contours, take on
colors that are consistent with those of the filled regions;
for example, a contour line on the boundary between two
regions will have a color intermediate between those
of the regions.

### Function: _process_linewidths(self, linewidths)

### Function: _process_linestyles(self, linestyles)

### Function: _find_nearest_contour(self, xy, indices)

**Description:** Find the point in the unfilled contour plot that is closest (in screen
space) to point *xy*.

Parameters
----------
xy : tuple[float, float]
    The reference point (in screen space).
indices : list of int or None, default: None
    Indices of contour levels to consider.  If None (the default), all levels
    are considered.

Returns
-------
idx_level_min : int
    The index of the contour level closest to *xy*.
idx_vtx_min : int
    The index of the `.Path` segment closest to *xy* (at that level).
proj : (float, float)
    The point in the contour plot closest to *xy*.

### Function: find_nearest_contour(self, x, y, indices, pixel)

**Description:** Find the point in the contour plot that is closest to ``(x, y)``.

This method does not support filled contours.

Parameters
----------
x, y : float
    The reference point.
indices : list of int or None, default: None
    Indices of contour levels to consider.  If None (the default), all
    levels are considered.
pixel : bool, default: True
    If *True*, measure distance in pixel (screen) space, which is
    useful for manual contour labeling; else, measure distance in axes
    space.

Returns
-------
path : int
    The index of the path that is closest to ``(x, y)``.  Each path corresponds
    to one contour level.
subpath : int
    The index within that closest path of the subpath that is closest to
    ``(x, y)``.  Each subpath corresponds to one unbroken contour line.
index : int
    The index of the vertices within that subpath that are closest to
    ``(x, y)``.
xmin, ymin : float
    The point in the contour plot that is closest to ``(x, y)``.
d2 : float
    The squared distance from ``(xmin, ymin)`` to ``(x, y)``.

### Function: draw(self, renderer)

### Function: _process_args(self)

**Description:** Process args and kwargs.

### Function: _contour_args(self, args, kwargs)

### Function: _check_xyz(self, x, y, z, kwargs)

**Description:** Check that the shapes of the input arrays match; if x and y are 1D,
convert them to 2D using meshgrid.

### Function: _initialize_x_y(self, z)

**Description:** Return X, Y arrays such that contour(Z) will match imshow(Z)
if origin is not None.
The center of pixel Z[i, j] depends on origin:
if origin is None, x = j, y = i;
if origin is 'lower', x = j + 0.5, y = i + 0.5;
if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
If extent is not None, x and y will be scaled to match,
as in imshow.
If origin is None and extent is not None, then extent
will give the minimum and maximum values of x and y.

### Function: interp_vec(x, xp, fp)
