## AI Summary

A file named grid_finder.py.


### Function: _find_line_box_crossings(xys, bbox)

**Description:** Find the points where a polyline crosses a bbox, and the crossing angles.

Parameters
----------
xys : (N, 2) array
    The polyline coordinates.
bbox : `.Bbox`
    The bounding box.

Returns
-------
list of ((float, float), float)
    Four separate lists of crossings, for the left, right, bottom, and top
    sides of the bbox, respectively.  For each list, the entries are the
    ``((x, y), ccw_angle_in_degrees)`` of the crossing, where an angle of 0
    means that the polyline is moving to the right at the crossing point.

    The entries are computed by linearly interpolating at each crossing
    between the nearest points on either side of the bbox edges.

## Class: ExtremeFinderSimple

**Description:** A helper class to figure out the range of grid lines that need to be drawn.

## Class: _User2DTransform

**Description:** A transform defined by two user-set functions.

## Class: GridFinder

**Description:** Internal helper for `~.grid_helper_curvelinear.GridHelperCurveLinear`, with
the same constructor parameters; should not be directly instantiated.

## Class: MaxNLocator

## Class: FixedLocator

## Class: FormatterPrettyPrint

## Class: DictFormatter

### Function: __init__(self, nx, ny)

**Description:** Parameters
----------
nx, ny : int
    The number of samples in each direction.

### Function: __call__(self, transform_xy, x1, y1, x2, y2)

**Description:** Compute an approximation of the bounding box obtained by applying
*transform_xy* to the box delimited by ``(x1, y1, x2, y2)``.

The intended use is to have ``(x1, y1, x2, y2)`` in axes coordinates,
and have *transform_xy* be the transform from axes coordinates to data
coordinates; this method then returns the range of data coordinates
that span the actual axes.

The computation is done by sampling ``nx * ny`` equispaced points in
the ``(x1, y1, x2, y2)`` box and finding the resulting points with
extremal coordinates; then adding some padding to take into account the
finite sampling.

As each sampling step covers a relative range of *1/nx* or *1/ny*,
the padding is computed by expanding the span covered by the extremal
coordinates by these fractions.

### Function: _add_pad(self, x_min, x_max, y_min, y_max)

**Description:** Perform the padding mentioned in `__call__`.

### Function: __init__(self, forward, backward)

**Description:** Parameters
----------
forward, backward : callable
    The forward and backward transforms, taking ``x`` and ``y`` as
    separate arguments and returning ``(tr_x, tr_y)``.

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, transform, extreme_finder, grid_locator1, grid_locator2, tick_formatter1, tick_formatter2)

### Function: _format_ticks(self, idx, direction, factor, levels)

**Description:** Helper to support both standard formatters (inheriting from
`.mticker.Formatter`) and axisartist-specific ones; should be called instead of
directly calling ``self.tick_formatter1`` and ``self.tick_formatter2``.  This
method should be considered as a temporary workaround which will be removed in
the future at the same time as axisartist-specific formatters.

### Function: get_grid_info(self, x1, y1, x2, y2)

**Description:** lon_values, lat_values : list of grid values. if integer is given,
                   rough number of grids in each direction.

### Function: _get_raw_grid_lines(self, lon_values, lat_values, lon_min, lon_max, lat_min, lat_max)

### Function: set_transform(self, aux_trans)

### Function: get_transform(self)

### Function: transform_xy(self, x, y)

### Function: inv_transform_xy(self, x, y)

### Function: update(self)

### Function: __init__(self, nbins, steps, trim, integer, symmetric, prune)

### Function: __call__(self, v1, v2)

### Function: __init__(self, locs)

### Function: __call__(self, v1, v2)

### Function: __init__(self, useMathText)

### Function: __call__(self, direction, factor, values)

### Function: __init__(self, format_dict, formatter)

**Description:** format_dict : dictionary for format strings to be used.
formatter : fall-back formatter

### Function: __call__(self, direction, factor, values)

**Description:** factor is ignored if value is found in the dictionary
