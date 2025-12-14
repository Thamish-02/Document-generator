## AI Summary

A file named floating_axes.py.


## Class: FloatingAxisArtistHelper

## Class: FixedAxisArtistHelper

## Class: ExtremeFinderFixed

## Class: GridHelperCurveLinear

## Class: FloatingAxesBase

### Function: __init__(self, grid_helper, side, nth_coord_ticks)

**Description:** nth_coord = along which coordinate value varies.
 nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis

### Function: update_lim(self, axes)

### Function: get_tick_iterators(self, axes)

**Description:** tick_loc, tick_angle, tick_label, (optionally) tick_label

### Function: get_line(self, axes)

### Function: __init__(self, extremes)

**Description:** This subclass always returns the same bounding box.

Parameters
----------
extremes : (float, float, float, float)
    The bounding box that this helper always returns.

### Function: __call__(self, transform_xy, x1, y1, x2, y2)

### Function: __init__(self, aux_trans, extremes, grid_locator1, grid_locator2, tick_formatter1, tick_formatter2)

### Function: new_fixed_axis(self, loc, nth_coord, axis_direction, offset, axes)

### Function: _update_grid(self, x1, y1, x2, y2)

### Function: get_gridlines(self, which, axis)

### Function: __init__(self)

### Function: _gen_axes_patch(self)

### Function: clear(self)

### Function: adjust_axes_lim(self)

### Function: trf_xy(x, y)

### Function: f1()
