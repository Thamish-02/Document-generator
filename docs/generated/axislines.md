## AI Summary

A file named axislines.py.


## Class: _AxisArtistHelperBase

**Description:** Base class for axis helper.

Subclasses should define the methods listed below.  The *axes*
argument will be the ``.axes`` attribute of the caller artist. ::

    # Construct the spine.

    def get_line_transform(self, axes):
        return transform

    def get_line(self, axes):
        return path

    # Construct the label.

    def get_axislabel_transform(self, axes):
        return transform

    def get_axislabel_pos_angle(self, axes):
        return (x, y), angle

    # Construct the ticks.

    def get_tick_transform(self, axes):
        return transform

    def get_tick_iterators(self, axes):
        # A pair of iterables (one for major ticks, one for minor ticks)
        # that yield (tick_position, tick_angle, tick_label).
        return iter_major, iter_minor

## Class: _FixedAxisArtistHelperBase

**Description:** Helper class for a fixed (in the axes coordinate) axis.

## Class: _FloatingAxisArtistHelperBase

## Class: FixedAxisArtistHelperRectilinear

## Class: FloatingAxisArtistHelperRectilinear

## Class: AxisArtistHelper

## Class: AxisArtistHelperRectlinear

## Class: GridHelperBase

## Class: GridHelperRectlinear

## Class: Axes

## Class: AxesZero

### Function: __init__(self, nth_coord)

### Function: update_lim(self, axes)

### Function: get_nth_coord(self)

### Function: _to_xy(self, values, const)

**Description:** Create a (*values.shape, 2)-shape array representing (x, y) pairs.

The other coordinate is filled with the constant *const*.

Example::

    >>> self.nth_coord = 0
    >>> self._to_xy([1, 2, 3], const=0)
    array([[1, 0],
           [2, 0],
           [3, 0]])

### Function: __init__(self, loc, nth_coord)

**Description:** ``nth_coord = 0``: x-axis; ``nth_coord = 1``: y-axis.

### Function: get_line(self, axes)

### Function: get_line_transform(self, axes)

### Function: get_axislabel_transform(self, axes)

### Function: get_axislabel_pos_angle(self, axes)

**Description:** Return the label reference position in transAxes.

get_label_transform() returns a transform of (transAxes+offset)

### Function: get_tick_transform(self, axes)

### Function: __init__(self, nth_coord, value)

### Function: get_line(self, axes)

### Function: __init__(self, axes, loc, nth_coord)

**Description:** nth_coord = along which coordinate value varies
in 2D, nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis

### Function: get_tick_iterators(self, axes)

**Description:** tick_loc, tick_angle, tick_label

### Function: __init__(self, axes, nth_coord, passingthrough_point, axis_direction)

### Function: get_line(self, axes)

### Function: get_line_transform(self, axes)

### Function: get_axislabel_transform(self, axes)

### Function: get_axislabel_pos_angle(self, axes)

**Description:** Return the label reference position in transAxes.

get_label_transform() returns a transform of (transAxes+offset)

### Function: get_tick_transform(self, axes)

### Function: get_tick_iterators(self, axes)

**Description:** tick_loc, tick_angle, tick_label

### Function: __init__(self)

### Function: update_lim(self, axes)

### Function: _update_grid(self, x1, y1, x2, y2)

**Description:** Cache relevant computations when the axes limits have changed.

### Function: get_gridlines(self, which, axis)

**Description:** Return list of grid lines as a list of paths (list of points).

Parameters
----------
which : {"both", "major", "minor"}
axis : {"both", "x", "y"}

### Function: __init__(self, axes)

### Function: new_fixed_axis(self, loc, nth_coord, axis_direction, offset, axes)

### Function: new_floating_axis(self, nth_coord, value, axis_direction, axes)

### Function: get_gridlines(self, which, axis)

**Description:** Return list of gridline coordinates in data coordinates.

Parameters
----------
which : {"both", "major", "minor"}
axis : {"both", "x", "y"}

### Function: __init__(self)

### Function: toggle_axisline(self, b)

### Function: axis(self)

### Function: clear(self)

### Function: get_grid_helper(self)

### Function: grid(self, visible, which, axis)

**Description:** Toggle the gridlines, and optionally set the properties of the lines.

### Function: get_children(self)

### Function: new_fixed_axis(self, loc, offset)

### Function: new_floating_axis(self, nth_coord, value, axis_direction)

### Function: clear(self)

### Function: _f(locs, labels)

### Function: _f(locs, labels)
