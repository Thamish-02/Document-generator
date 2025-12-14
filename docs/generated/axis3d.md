## AI Summary

A file named axis3d.py.


### Function: _move_from_center(coord, centers, deltas, axmask)

**Description:** For each coordinate where *axmask* is True, move *coord* away from
*centers* by *deltas*.

### Function: _tick_update_position(tick, tickxs, tickys, labelpos)

**Description:** Update tick line and label position and style.

## Class: Axis

**Description:** An Axis class for the 3D plots.

## Class: XAxis

## Class: YAxis

## Class: ZAxis

### Function: _old_init(self, adir, v_intervalx, d_intervalx, axes)

### Function: _new_init(self, axes)

### Function: __init__(self)

### Function: _init3d(self)

### Function: init3d(self)

### Function: get_major_ticks(self, numticks)

### Function: get_minor_ticks(self, numticks)

### Function: set_ticks_position(self, position)

**Description:** Set the ticks position.

Parameters
----------
position : {'lower', 'upper', 'both', 'default', 'none'}
    The position of the bolded axis lines, ticks, and tick labels.

### Function: get_ticks_position(self)

**Description:** Get the ticks position.

Returns
-------
str : {'lower', 'upper', 'both', 'default', 'none'}
    The position of the bolded axis lines, ticks, and tick labels.

### Function: set_label_position(self, position)

**Description:** Set the label position.

Parameters
----------
position : {'lower', 'upper', 'both', 'default', 'none'}
    The position of the axis label.

### Function: get_label_position(self)

**Description:** Get the label position.

Returns
-------
str : {'lower', 'upper', 'both', 'default', 'none'}
    The position of the axis label.

### Function: set_pane_color(self, color, alpha)

**Description:** Set pane color.

Parameters
----------
color : :mpltype:`color`
    Color for axis pane.
alpha : float, optional
    Alpha value for axis pane. If None, base it on *color*.

### Function: set_rotate_label(self, val)

**Description:** Whether to rotate the axis label: True, False or None.
If set to None the label will be rotated if longer than 4 chars.

### Function: get_rotate_label(self, text)

### Function: _get_coord_info(self)

### Function: _calc_centers_deltas(self, maxs, mins)

### Function: _get_axis_line_edge_points(self, minmax, maxmin, position)

**Description:** Get the edge points for the black bolded axis line.

### Function: _get_all_axis_line_edge_points(self, minmax, maxmin, axis_position)

### Function: _get_tickdir(self, position)

**Description:** Get the direction of the tick.

Parameters
----------
position : str, optional : {'upper', 'lower', 'default'}
    The position of the axis.

Returns
-------
tickdir : int
    Index which indicates which coordinate the tick line will
    align with.

### Function: active_pane(self)

### Function: draw_pane(self, renderer)

**Description:** Draw pane.

Parameters
----------
renderer : `~matplotlib.backend_bases.RendererBase` subclass

### Function: _axmask(self)

### Function: _draw_ticks(self, renderer, edgep1, centers, deltas, highs, deltas_per_point, pos)

### Function: _draw_offset_text(self, renderer, edgep1, edgep2, labeldeltas, centers, highs, pep, dx, dy)

### Function: _draw_labels(self, renderer, edgep1, edgep2, labeldeltas, centers, dx, dy)

### Function: draw(self, renderer)

### Function: draw_grid(self, renderer)

### Function: get_tightbbox(self, renderer)
