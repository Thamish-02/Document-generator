## AI Summary

A file named axis_artist.py.


## Class: AttributeCopier

## Class: Ticks

**Description:** Ticks are derived from `.Line2D`, and note that ticks themselves
are markers. Thus, you should use set_mec, set_mew, etc.

To change the tick size (length), you need to use
`set_ticksize`. To change the direction of the ticks (ticks are
in opposite direction of ticklabels by default), use
``set_tick_out(False)``

## Class: LabelBase

**Description:** A base class for `.AxisLabel` and `.TickLabels`. The position and
angle of the text are calculated by the offset_ref_angle,
text_ref_angle, and offset_radius attributes.

## Class: AxisLabel

**Description:** Axis label. Derived from `.Text`. The position of the text is updated
in the fly, so changing text position has no effect. Otherwise, the
properties can be changed as a normal `.Text`.

To change the pad between tick labels and axis label, use `set_pad`.

## Class: TickLabels

**Description:** Tick labels. While derived from `.Text`, this single artist draws all
ticklabels. As in `.AxisLabel`, the position of the text is updated
in the fly, so changing text position has no effect. Otherwise,
the properties can be changed as a normal `.Text`. Unlike the
ticklabels of the mainline Matplotlib, properties of a single
ticklabel alone cannot be modified.

To change the pad between ticks and ticklabels, use `~.AxisLabel.set_pad`.

## Class: GridlinesCollection

## Class: AxisArtist

**Description:** An artist which draws axis (a line along which the n-th axes coord
is constant) line, ticks, tick labels, and axis label.

### Function: get_ref_artist(self)

**Description:** Return the underlying artist that actually defines some properties
(e.g., color) of this artist.

### Function: get_attribute_from_ref_artist(self, attr_name)

### Function: __init__(self, ticksize, tick_out)

### Function: get_ref_artist(self)

### Function: set_color(self, color)

### Function: get_color(self)

### Function: get_markeredgecolor(self)

### Function: get_markeredgewidth(self)

### Function: set_tick_out(self, b)

**Description:** Set whether ticks are drawn inside or outside the axes.

### Function: get_tick_out(self)

**Description:** Return whether ticks are drawn inside or outside the axes.

### Function: set_ticksize(self, ticksize)

**Description:** Set length of the ticks in points.

### Function: get_ticksize(self)

**Description:** Return length of the ticks in points.

### Function: set_locs_angles(self, locs_angles)

### Function: draw(self, renderer)

### Function: __init__(self)

### Function: _text_ref_angle(self)

### Function: _offset_ref_angle(self)

### Function: draw(self, renderer)

### Function: get_window_extent(self, renderer)

### Function: __init__(self)

### Function: set_pad(self, pad)

**Description:** Set the internal pad in points.

The actual pad will be the sum of the internal pad and the
external pad (the latter is set automatically by the `.AxisArtist`).

Parameters
----------
pad : float
    The internal pad in points.

### Function: get_pad(self)

**Description:** Return the internal pad in points.

See `.set_pad` for more details.

### Function: get_ref_artist(self)

### Function: get_text(self)

### Function: set_default_alignment(self, d)

**Description:** Set the default alignment. See `set_axis_direction` for details.

Parameters
----------
d : {"left", "bottom", "right", "top"}

### Function: set_default_angle(self, d)

**Description:** Set the default angle. See `set_axis_direction` for details.

Parameters
----------
d : {"left", "bottom", "right", "top"}

### Function: set_axis_direction(self, d)

**Description:** Adjust the text angle and text alignment of axis label
according to the matplotlib convention.

=====================    ========== ========= ========== ==========
Property                 left       bottom    right      top
=====================    ========== ========= ========== ==========
axislabel angle          180        0         0          180
axislabel va             center     top       center     bottom
axislabel ha             right      center    right      center
=====================    ========== ========= ========== ==========

Note that the text angles are actually relative to (90 + angle
of the direction to the ticklabel), which gives 0 for bottom
axis.

Parameters
----------
d : {"left", "bottom", "right", "top"}

### Function: get_color(self)

### Function: draw(self, renderer)

### Function: get_window_extent(self, renderer)

### Function: __init__(self)

### Function: get_ref_artist(self)

### Function: set_axis_direction(self, label_direction)

**Description:** Adjust the text angle and text alignment of ticklabels
according to the Matplotlib convention.

The *label_direction* must be one of [left, right, bottom, top].

=====================    ========== ========= ========== ==========
Property                 left       bottom    right      top
=====================    ========== ========= ========== ==========
ticklabel angle          90         0         -90        180
ticklabel va             center     baseline  center     baseline
ticklabel ha             right      center    right      center
=====================    ========== ========= ========== ==========

Note that the text angles are actually relative to (90 + angle
of the direction to the ticklabel), which gives 0 for bottom
axis.

Parameters
----------
label_direction : {"left", "bottom", "right", "top"}

### Function: invert_axis_direction(self)

### Function: _get_ticklabels_offsets(self, renderer, label_direction)

**Description:** Calculate the ticklabel offsets from the tick and their total heights.

The offset only takes account the offset due to the vertical alignment
of the ticklabels: if axis direction is bottom and va is 'top', it will
return 0; if va is 'baseline', it will return (height-descent).

### Function: draw(self, renderer)

### Function: set_locs_angles_labels(self, locs_angles_labels)

### Function: get_window_extents(self, renderer)

### Function: get_texts_widths_heights_descents(self, renderer)

**Description:** Return a list of ``(width, height, descent)`` tuples for ticklabels.

Empty labels are left out.

### Function: __init__(self)

**Description:** Collection of grid lines.

Parameters
----------
which : {"major", "minor"}
    Which grid to consider.
axis : {"both", "x", "y"}
    Which axis to consider.
*args, **kwargs
    Passed to `.LineCollection`.

### Function: set_which(self, which)

**Description:** Select major or minor grid lines.

Parameters
----------
which : {"major", "minor"}

### Function: set_axis(self, axis)

**Description:** Select axis.

Parameters
----------
axis : {"both", "x", "y"}

### Function: set_grid_helper(self, grid_helper)

**Description:** Set grid helper.

Parameters
----------
grid_helper : `.GridHelperBase` subclass

### Function: draw(self, renderer)

### Function: LABELPAD(self)

### Function: LABELPAD(self, v)

### Function: __init__(self, axes, helper, offset, axis_direction)

**Description:** Parameters
----------
axes : `mpl_toolkits.axisartist.axislines.Axes`
helper : `~mpl_toolkits.axisartist.axislines.AxisArtistHelper`

### Function: set_axis_direction(self, axis_direction)

**Description:** Adjust the direction, text angle, and text alignment of tick labels
and axis labels following the Matplotlib convention for the rectangle
axes.

The *axis_direction* must be one of [left, right, bottom, top].

=====================    ========== ========= ========== ==========
Property                 left       bottom    right      top
=====================    ========== ========= ========== ==========
ticklabel direction      "-"        "+"       "+"        "-"
axislabel direction      "-"        "+"       "+"        "-"
ticklabel angle          90         0         -90        180
ticklabel va             center     baseline  center     baseline
ticklabel ha             right      center    right      center
axislabel angle          180        0         0          180
axislabel va             center     top       center     bottom
axislabel ha             right      center    right      center
=====================    ========== ========= ========== ==========

Note that the direction "+" and "-" are relative to the direction of
the increasing coordinate. Also, the text angles are actually
relative to (90 + angle of the direction to the ticklabel),
which gives 0 for bottom axis.

Parameters
----------
axis_direction : {"left", "bottom", "right", "top"}

### Function: set_ticklabel_direction(self, tick_direction)

**Description:** Adjust the direction of the tick labels.

Note that the *tick_direction*\s '+' and '-' are relative to the
direction of the increasing coordinate.

Parameters
----------
tick_direction : {"+", "-"}

### Function: invert_ticklabel_direction(self)

### Function: set_axislabel_direction(self, label_direction)

**Description:** Adjust the direction of the axis label.

Note that the *label_direction*\s '+' and '-' are relative to the
direction of the increasing coordinate.

Parameters
----------
label_direction : {"+", "-"}

### Function: get_transform(self)

### Function: get_helper(self)

**Description:** Return axis artist helper instance.

### Function: set_axisline_style(self, axisline_style)

**Description:** Set the axisline style.

The new style is completely defined by the passed attributes. Existing
style attributes are forgotten.

Parameters
----------
axisline_style : str or None
    The line style, e.g. '->', optionally followed by a comma-separated
    list of attributes. Alternatively, the attributes can be provided
    as keywords.

    If *None* this returns a string containing the available styles.

Examples
--------
The following two commands are equal:

>>> set_axisline_style("->,size=1.5")
>>> set_axisline_style("->", size=1.5)

### Function: get_axisline_style(self)

**Description:** Return the current axisline style.

### Function: _init_line(self)

**Description:** Initialize the *line* artist that is responsible to draw the axis line.

### Function: _draw_line(self, renderer)

### Function: _init_ticks(self)

### Function: _get_tick_info(self, tick_iter)

**Description:** Return a pair of:

- list of locs and angles for ticks
- list of locs, angles and labels for ticklabels.

### Function: _update_ticks(self, renderer)

### Function: _draw_ticks(self, renderer)

### Function: _init_offsetText(self, direction)

### Function: _update_offsetText(self)

### Function: _draw_offsetText(self, renderer)

### Function: _init_label(self)

### Function: _update_label(self, renderer)

### Function: _draw_label(self, renderer)

### Function: set_label(self, s)

### Function: get_tightbbox(self, renderer)

### Function: draw(self, renderer)

### Function: toggle(self, all, ticks, ticklabels, label)

**Description:** Toggle visibility of ticks, ticklabels, and (axis) label.
To turn all off, ::

  axis.toggle(all=False)

To turn all off but ticks on ::

  axis.toggle(all=False, ticks=True)

To turn all on but (axis) label off ::

  axis.toggle(all=True, label=False)
