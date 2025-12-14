## AI Summary

A file named grid_helper_curvelinear.py.


### Function: _value_and_jacobian(func, xs, ys, xlims, ylims)

**Description:** Compute *func* and its derivatives along x and y at positions *xs*, *ys*,
while ensuring that finite difference calculations don't try to evaluate
values outside of *xlims*, *ylims*.

## Class: FixedAxisArtistHelper

**Description:** Helper class for a fixed axis.

## Class: FloatingAxisArtistHelper

## Class: GridHelperCurveLinear

### Function: __init__(self, grid_helper, side, nth_coord_ticks)

**Description:** nth_coord = along which coordinate value varies.
 nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis

### Function: update_lim(self, axes)

### Function: get_tick_transform(self, axes)

### Function: get_tick_iterators(self, axes)

**Description:** tick_loc, tick_angle, tick_label

### Function: __init__(self, grid_helper, nth_coord, value, axis_direction)

**Description:** nth_coord = along which coordinate value varies.
 nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis

### Function: set_extremes(self, e1, e2)

### Function: update_lim(self, axes)

### Function: get_axislabel_transform(self, axes)

### Function: get_axislabel_pos_angle(self, axes)

### Function: get_tick_transform(self, axes)

### Function: get_tick_iterators(self, axes)

**Description:** tick_loc, tick_angle, tick_label, (optionally) tick_label

### Function: get_line_transform(self, axes)

### Function: get_line(self, axes)

### Function: __init__(self, aux_trans, extreme_finder, grid_locator1, grid_locator2, tick_formatter1, tick_formatter2)

**Description:** Parameters
----------
aux_trans : `.Transform` or tuple[Callable, Callable]
    The transform from curved coordinates to rectilinear coordinate:
    either a `.Transform` instance (which provides also its inverse),
    or a pair of callables ``(trans, inv_trans)`` that define the
    transform and its inverse.  The callables should have signature::

        x_rect, y_rect = trans(x_curved, y_curved)
        x_curved, y_curved = inv_trans(x_rect, y_rect)

extreme_finder

grid_locator1, grid_locator2
    Grid locators for each axis.

tick_formatter1, tick_formatter2
    Tick formatters for each axis.

### Function: update_grid_finder(self, aux_trans)

### Function: new_fixed_axis(self, loc, nth_coord, axis_direction, offset, axes)

### Function: new_floating_axis(self, nth_coord, value, axes, axis_direction)

### Function: _update_grid(self, x1, y1, x2, y2)

### Function: get_gridlines(self, which, axis)

### Function: get_tick_iterator(self, nth_coord, axis_side, minor)

### Function: iter_major()

### Function: trf_xy(x, y)

### Function: trf_xy(x, y)

### Function: iter_major()
