## AI Summary

A file named axis.py.


## Class: Tick

**Description:** Abstract base class for the axis ticks, grid lines and labels.

Ticks mark a position on an Axis. They contain two lines as markers and
two labels; one each for the bottom and top positions (in case of an
`.XAxis`) or for the left and right positions (in case of a `.YAxis`).

Attributes
----------
tick1line : `~matplotlib.lines.Line2D`
    The left/bottom tick marker.
tick2line : `~matplotlib.lines.Line2D`
    The right/top tick marker.
gridline : `~matplotlib.lines.Line2D`
    The grid line associated with the label position.
label1 : `~matplotlib.text.Text`
    The left/bottom tick label.
label2 : `~matplotlib.text.Text`
    The right/top tick label.

## Class: XTick

**Description:** Contains all the Artists needed to make an x tick - the tick line,
the label text and the grid line

## Class: YTick

**Description:** Contains all the Artists needed to make a Y tick - the tick line,
the label text and the grid line

## Class: Ticker

**Description:** A container for the objects defining tick position and format.

Attributes
----------
locator : `~matplotlib.ticker.Locator` subclass
    Determines the positions of the ticks.
formatter : `~matplotlib.ticker.Formatter` subclass
    Determines the format of the tick labels.

## Class: _LazyTickList

**Description:** A descriptor for lazy instantiation of tick lists.

See comment above definition of the ``majorTicks`` and ``minorTicks``
attributes.

## Class: Axis

**Description:** Base class for `.XAxis` and `.YAxis`.

Attributes
----------
isDefault_label : bool

axes : `~matplotlib.axes.Axes`
    The `~.axes.Axes` to which the Axis belongs.
major : `~matplotlib.axis.Ticker`
    Determines the major tick positions and their label format.
minor : `~matplotlib.axis.Ticker`
    Determines the minor tick positions and their label format.
callbacks : `~matplotlib.cbook.CallbackRegistry`

label : `~matplotlib.text.Text`
    The axis label.
labelpad : float
    The distance between the axis label and the tick labels.
    Defaults to :rc:`axes.labelpad`.
offsetText : `~matplotlib.text.Text`
    A `.Text` object containing the data offset of the ticks (if any).
pickradius : float
    The acceptance radius for containment tests. See also `.Axis.contains`.
majorTicks : list of `.Tick`
    The major ticks.

    .. warning::

        Ticks are not guaranteed to be persistent. Various operations
        can create, delete and modify the Tick instances. There is an
        imminent risk that changes to individual ticks will not
        survive if you work on the figure further (including also
        panning/zooming on a displayed figure).

        Working on the individual ticks is a method of last resort.
        Use `.set_tick_params` instead if possible.

minorTicks : list of `.Tick`
    The minor ticks.

### Function: _make_getset_interval(method_name, lim_name, attr_name)

**Description:** Helper to generate ``get_{data,view}_interval`` and
``set_{data,view}_interval`` implementations.

## Class: XAxis

## Class: YAxis

### Function: __init__(self, axes, loc)

**Description:** bbox is the Bound2D bounding box in display coords of the Axes
loc is the tick location in data coords
size is the tick size in points

### Function: _set_labelrotation(self, labelrotation)

### Function: _pad(self)

### Function: _apply_tickdir(self, tickdir)

**Description:** Set tick direction.  Valid values are 'out', 'in', 'inout'.

### Function: get_tickdir(self)

### Function: get_tick_padding(self)

**Description:** Get the length of the tick outside of the Axes.

### Function: get_children(self)

### Function: set_clip_path(self, path, transform)

### Function: contains(self, mouseevent)

**Description:** Test whether the mouse event occurred in the Tick marks.

This function always returns false.  It is more useful to test if the
axis as a whole contains the mouse rather than the set of tick marks.

### Function: set_pad(self, val)

**Description:** Set the tick label pad in points

Parameters
----------
val : float

### Function: get_pad(self)

**Description:** Get the value of the tick label pad in points.

### Function: get_loc(self)

**Description:** Return the tick location (data coords) as a scalar.

### Function: draw(self, renderer)

### Function: set_url(self, url)

**Description:** Set the url of label1 and label2.

Parameters
----------
url : str

### Function: _set_artist_props(self, a)

### Function: get_view_interval(self)

**Description:** Return the view limits ``(min, max)`` of the axis the tick belongs to.

### Function: _apply_params(self)

### Function: update_position(self, loc)

**Description:** Set the location of tick in data coords with scalar *loc*.

### Function: _get_text1_transform(self)

### Function: _get_text2_transform(self)

### Function: __init__(self)

### Function: _get_text1_transform(self)

### Function: _get_text2_transform(self)

### Function: _apply_tickdir(self, tickdir)

### Function: update_position(self, loc)

**Description:** Set the location of tick in data coords with scalar *loc*.

### Function: get_view_interval(self)

### Function: __init__(self)

### Function: _get_text1_transform(self)

### Function: _get_text2_transform(self)

### Function: _apply_tickdir(self, tickdir)

### Function: update_position(self, loc)

**Description:** Set the location of tick in data coords with scalar *loc*.

### Function: get_view_interval(self)

### Function: __init__(self)

### Function: locator(self)

### Function: locator(self, locator)

### Function: formatter(self)

### Function: formatter(self, formatter)

### Function: __init__(self, major)

### Function: __get__(self, instance, owner)

### Function: __str__(self)

### Function: __init__(self, axes)

**Description:** Parameters
----------
axes : `~matplotlib.axes.Axes`
    The `~.axes.Axes` to which the created Axis belongs.
pickradius : float
    The acceptance radius for containment tests. See also
    `.Axis.contains`.
clear : bool, default: True
    Whether to clear the Axis on creation. This is not required, e.g.,  when
    creating an Axis as part of an Axes, as ``Axes.clear`` will call
    ``Axis.clear``.
    .. versionadded:: 3.8

### Function: isDefault_majloc(self)

### Function: isDefault_majloc(self, value)

### Function: isDefault_majfmt(self)

### Function: isDefault_majfmt(self, value)

### Function: isDefault_minloc(self)

### Function: isDefault_minloc(self, value)

### Function: isDefault_minfmt(self)

### Function: isDefault_minfmt(self, value)

### Function: _get_shared_axes(self)

**Description:** Return Grouper of shared Axes for current axis.

### Function: _get_shared_axis(self)

**Description:** Return list of shared axis for current axis.

### Function: _get_axis_name(self)

**Description:** Return the axis name.

### Function: get_remove_overlapping_locs(self)

### Function: set_remove_overlapping_locs(self, val)

### Function: set_label_coords(self, x, y, transform)

**Description:** Set the coordinates of the label.

By default, the x coordinate of the y label and the y coordinate of the
x label are determined by the tick label bounding boxes, but this can
lead to poor alignment of multiple labels if there are multiple Axes.

You can also specify the coordinate system of the label with the
transform.  If None, the default coordinate system will be the axes
coordinate system: (0, 0) is bottom left, (0.5, 0.5) is center, etc.

### Function: get_transform(self)

**Description:** Return the transform used in the Axis' scale

### Function: get_scale(self)

**Description:** Return this Axis' scale (as a str).

### Function: _set_scale(self, value)

### Function: _set_axes_scale(self, value)

**Description:** Set this Axis' scale.

Parameters
----------
value : str or `.ScaleBase`
    The axis scale type to apply.  Valid string values are the names of scale
    classes ("linear", "log", "function",...).  These may be the names of any
    of the :ref:`built-in scales<builtin_scales>` or of any custom scales
    registered using `matplotlib.scale.register_scale`.

**kwargs
    If *value* is a string, keywords are passed to the instantiation method of
    the respective class.

### Function: limit_range_for_scale(self, vmin, vmax)

**Description:** Return the range *vmin*, *vmax*, restricted to the domain supported by the
current scale.

### Function: _get_autoscale_on(self)

**Description:** Return whether this Axis is autoscaled.

### Function: _set_autoscale_on(self, b)

**Description:** Set whether this Axis is autoscaled when drawing or by `.Axes.autoscale_view`.

If b is None, then the value is not changed.

Parameters
----------
b : bool

### Function: get_children(self)

### Function: _reset_major_tick_kw(self)

### Function: _reset_minor_tick_kw(self)

### Function: clear(self)

**Description:** Clear the axis.

This resets axis properties to their default values:

- the label
- the scale
- locators, formatters and ticks
- major and minor grid
- units
- registered callbacks

### Function: reset_ticks(self)

**Description:** Re-initialize the major and minor Tick lists.

Each list starts with a single fresh Tick.

### Function: minorticks_on(self)

**Description:** Display default minor ticks on the Axis, depending on the scale
(`~.axis.Axis.get_scale`).

Scales use specific minor locators:

- log: `~.LogLocator`
- symlog: `~.SymmetricalLogLocator`
- asinh: `~.AsinhLocator`
- logit: `~.LogitLocator`
- default: `~.AutoMinorLocator`

Displaying minor ticks may reduce performance; you may turn them off
using `minorticks_off()` if drawing speed is a problem.

### Function: minorticks_off(self)

**Description:** Remove minor ticks from the Axis.

### Function: set_tick_params(self, which, reset)

**Description:** Set appearance parameters for ticks, ticklabels, and gridlines.

For documentation of keyword arguments, see
:meth:`matplotlib.axes.Axes.tick_params`.

See Also
--------
.Axis.get_tick_params
    View the current style settings for ticks, ticklabels, and
    gridlines.

### Function: get_tick_params(self, which)

**Description:** Get appearance parameters for ticks, ticklabels, and gridlines.

.. versionadded:: 3.7

Parameters
----------
which : {'major', 'minor'}, default: 'major'
    The group of ticks for which the parameters are retrieved.

Returns
-------
dict
    Properties for styling tick elements added to the axis.

Notes
-----
This method returns the appearance parameters for styling *new*
elements added to this axis and may be different from the values
on current elements if they were modified directly by the user
(e.g., via ``set_*`` methods on individual tick objects).

Examples
--------
::

    >>> ax.yaxis.set_tick_params(labelsize=30, labelcolor='red',
    ...                          direction='out', which='major')
    >>> ax.yaxis.get_tick_params(which='major')
    {'direction': 'out',
    'left': True,
    'right': False,
    'labelleft': True,
    'labelright': False,
    'gridOn': False,
    'labelsize': 30,
    'labelcolor': 'red'}
    >>> ax.yaxis.get_tick_params(which='minor')
    {'left': True,
    'right': False,
    'labelleft': True,
    'labelright': False,
    'gridOn': False}

### Function: _translate_tick_params(cls, kw, reverse)

**Description:** Translate the kwargs supported by `.Axis.set_tick_params` to kwargs
supported by `.Tick._apply_params`.

In particular, this maps axis specific names like 'top', 'left'
to the generic tick1, tick2 logic of the axis. Additionally, there
are some other name translations.

Returns a new dict of translated kwargs.

Note: Use reverse=True to translate from those supported by
`.Tick._apply_params` back to those supported by
`.Axis.set_tick_params`.

### Function: set_clip_path(self, path, transform)

### Function: get_view_interval(self)

**Description:** Return the ``(min, max)`` view limits of this axis.

### Function: set_view_interval(self, vmin, vmax, ignore)

**Description:** Set the axis view limits.  This method is for internal use; Matplotlib
users should typically use e.g. `~.Axes.set_xlim` or `~.Axes.set_ylim`.

If *ignore* is False (the default), this method will never reduce the
preexisting view limits, only expand them if *vmin* or *vmax* are not
within them.  Moreover, the order of *vmin* and *vmax* does not matter;
the orientation of the axis will not change.

If *ignore* is True, the view limits will be set exactly to ``(vmin,
vmax)`` in that order.

### Function: get_data_interval(self)

**Description:** Return the ``(min, max)`` data limits of this axis.

### Function: set_data_interval(self, vmin, vmax, ignore)

**Description:** Set the axis data limits.  This method is for internal use.

If *ignore* is False (the default), this method will never reduce the
preexisting data limits, only expand them if *vmin* or *vmax* are not
within them.  Moreover, the order of *vmin* and *vmax* does not matter;
the orientation of the axis will not change.

If *ignore* is True, the data limits will be set exactly to ``(vmin,
vmax)`` in that order.

### Function: get_inverted(self)

**Description:** Return whether this Axis is oriented in the "inverse" direction.

The "normal" direction is increasing to the right for the x-axis and to
the top for the y-axis; the "inverse" direction is increasing to the
left for the x-axis and to the bottom for the y-axis.

### Function: set_inverted(self, inverted)

**Description:** Set whether this Axis is oriented in the "inverse" direction.

The "normal" direction is increasing to the right for the x-axis and to
the top for the y-axis; the "inverse" direction is increasing to the
left for the x-axis and to the bottom for the y-axis.

### Function: set_default_intervals(self)

**Description:** Set the default limits for the axis data and view interval if they
have not been not mutated yet.

### Function: _set_lim(self, v0, v1)

**Description:** Set view limits.

This method is a helper for the Axes ``set_xlim``, ``set_ylim``, and
``set_zlim`` methods.

Parameters
----------
v0, v1 : float
    The view limits.  (Passing *v0* as a (low, high) pair is not
    supported; normalization must occur in the Axes setters.)
emit : bool, default: True
    Whether to notify observers of limit change.
auto : bool or None, default: False
    Whether to turn on autoscaling of the x-axis. True turns on, False
    turns off, None leaves unchanged.

### Function: _set_artist_props(self, a)

### Function: _update_ticks(self)

**Description:** Update ticks (position and labels) using the current data interval of
the axes.  Return the list of ticks that will be drawn.

### Function: _get_ticklabel_bboxes(self, ticks, renderer)

**Description:** Return lists of bboxes for ticks' label1's and label2's.

### Function: get_tightbbox(self, renderer)

**Description:** Return a bounding box that encloses the axis. It only accounts
tick labels, axis label, and offsetText.

If *for_layout_only* is True, then the width of the label (if this
is an x-axis) or the height of the label (if this is a y-axis) is
collapsed to near zero.  This allows tight/constrained_layout to ignore
too-long labels when doing their layout.

### Function: get_tick_padding(self)

### Function: draw(self, renderer)

### Function: get_gridlines(self)

**Description:** Return this Axis' grid lines as a list of `.Line2D`\s.

### Function: set_label(self, s)

**Description:** Assigning legend labels is not supported. Raises RuntimeError.

### Function: get_label(self)

**Description:** Return the axis label as a Text instance.

.. admonition:: Discouraged

   This overrides `.Artist.get_label`, which is for legend labels, with a new
   semantic. It is recommended to use the attribute ``Axis.label`` instead.

### Function: get_offset_text(self)

**Description:** Return the axis offsetText as a Text instance.

### Function: get_pickradius(self)

**Description:** Return the depth of the axis used by the picker.

### Function: get_majorticklabels(self)

**Description:** Return this Axis' major tick labels, as a list of `~.text.Text`.

### Function: get_minorticklabels(self)

**Description:** Return this Axis' minor tick labels, as a list of `~.text.Text`.

### Function: get_ticklabels(self, minor, which)

**Description:** Get this Axis' tick labels.

Parameters
----------
minor : bool
   Whether to return the minor or the major ticklabels.

which : None, ('minor', 'major', 'both')
   Overrides *minor*.

   Selects which ticklabels to return

Returns
-------
list of `~matplotlib.text.Text`

### Function: get_majorticklines(self)

**Description:** Return this Axis' major tick lines as a list of `.Line2D`\s.

### Function: get_minorticklines(self)

**Description:** Return this Axis' minor tick lines as a list of `.Line2D`\s.

### Function: get_ticklines(self, minor)

**Description:** Return this Axis' tick lines as a list of `.Line2D`\s.

### Function: get_majorticklocs(self)

**Description:** Return this Axis' major tick locations in data coordinates.

### Function: get_minorticklocs(self)

**Description:** Return this Axis' minor tick locations in data coordinates.

### Function: get_ticklocs(self)

**Description:** Return this Axis' tick locations in data coordinates.

The locations are not clipped to the current axis limits and hence
may contain locations that are not visible in the output.

Parameters
----------
minor : bool, default: False
    True to return the minor tick directions,
    False to return the major tick directions.

Returns
-------
array of tick locations

### Function: get_ticks_direction(self, minor)

**Description:** Return an array of this Axis' tick directions.

Parameters
----------
minor : bool, default: False
    True to return the minor tick directions,
    False to return the major tick directions.

Returns
-------
array of tick directions

### Function: _get_tick(self, major)

**Description:** Return the default tick instance.

### Function: _get_tick_label_size(self, axis_name)

**Description:** Return the text size of tick labels for this Axis.

This is a convenience function to avoid having to create a `Tick` in
`.get_tick_space`, since it is expensive.

### Function: _copy_tick_props(self, src, dest)

**Description:** Copy the properties from *src* tick to *dest* tick.

### Function: get_label_text(self)

**Description:** Get the text of the label.

### Function: get_major_locator(self)

**Description:** Get the locator of the major ticker.

### Function: get_minor_locator(self)

**Description:** Get the locator of the minor ticker.

### Function: get_major_formatter(self)

**Description:** Get the formatter of the major ticker.

### Function: get_minor_formatter(self)

**Description:** Get the formatter of the minor ticker.

### Function: get_major_ticks(self, numticks)

**Description:** Return the list of major `.Tick`\s.

.. warning::

    Ticks are not guaranteed to be persistent. Various operations
    can create, delete and modify the Tick instances. There is an
    imminent risk that changes to individual ticks will not
    survive if you work on the figure further (including also
    panning/zooming on a displayed figure).

    Working on the individual ticks is a method of last resort.
    Use `.set_tick_params` instead if possible.

### Function: get_minor_ticks(self, numticks)

**Description:** Return the list of minor `.Tick`\s.

.. warning::

    Ticks are not guaranteed to be persistent. Various operations
    can create, delete and modify the Tick instances. There is an
    imminent risk that changes to individual ticks will not
    survive if you work on the figure further (including also
    panning/zooming on a displayed figure).

    Working on the individual ticks is a method of last resort.
    Use `.set_tick_params` instead if possible.

### Function: grid(self, visible, which)

**Description:** Configure the grid lines.

Parameters
----------
visible : bool or None
    Whether to show the grid lines.  If any *kwargs* are supplied, it
    is assumed you want the grid on and *visible* will be set to True.

    If *visible* is *None* and there are no *kwargs*, this toggles the
    visibility of the lines.

which : {'major', 'minor', 'both'}
    The grid lines to apply the changes on.

**kwargs : `~matplotlib.lines.Line2D` properties
    Define the line properties of the grid, e.g.::

        grid(color='r', linestyle='-', linewidth=2)

### Function: update_units(self, data)

**Description:** Introspect *data* for units converter and update the
``axis.get_converter`` instance if necessary. Return *True*
if *data* is registered for unit conversion.

### Function: _update_axisinfo(self)

**Description:** Check the axis converter for the stored units to see if the
axis info needs to be updated.

### Function: have_units(self)

### Function: convert_units(self, x)

### Function: get_converter(self)

**Description:** Get the unit converter for axis.

Returns
-------
`~matplotlib.units.ConversionInterface` or None

### Function: set_converter(self, converter)

**Description:** Set the unit converter for axis.

Parameters
----------
converter : `~matplotlib.units.ConversionInterface`

### Function: _set_converter(self, converter)

### Function: set_units(self, u)

**Description:** Set the units for axis.

Parameters
----------
u : units tag

Notes
-----
The units of any shared axis will also be updated.

### Function: get_units(self)

**Description:** Return the units for axis.

### Function: set_label_text(self, label, fontdict)

**Description:** Set the text value of the axis label.

Parameters
----------
label : str
    Text string.
fontdict : dict
    Text properties.

    .. admonition:: Discouraged

       The use of *fontdict* is discouraged. Parameters should be passed as
       individual keyword arguments or using dictionary-unpacking
       ``set_label_text(..., **fontdict)``.

**kwargs
    Merged into fontdict.

### Function: set_major_formatter(self, formatter)

**Description:** Set the formatter of the major ticker.

In addition to a `~matplotlib.ticker.Formatter` instance,
this also accepts a ``str`` or function.

For a ``str`` a `~matplotlib.ticker.StrMethodFormatter` is used.
The field used for the value must be labeled ``'x'`` and the field used
for the position must be labeled ``'pos'``.
See the  `~matplotlib.ticker.StrMethodFormatter` documentation for
more information.

For a function, a `~matplotlib.ticker.FuncFormatter` is used.
The function must take two inputs (a tick value ``x`` and a
position ``pos``), and return a string containing the corresponding
tick label.
See the  `~matplotlib.ticker.FuncFormatter` documentation for
more information.

Parameters
----------
formatter : `~matplotlib.ticker.Formatter`, ``str``, or function

### Function: set_minor_formatter(self, formatter)

**Description:** Set the formatter of the minor ticker.

In addition to a `~matplotlib.ticker.Formatter` instance,
this also accepts a ``str`` or function.
See `.Axis.set_major_formatter` for more information.

Parameters
----------
formatter : `~matplotlib.ticker.Formatter`, ``str``, or function

### Function: _set_formatter(self, formatter, level)

### Function: set_major_locator(self, locator)

**Description:** Set the locator of the major ticker.

Parameters
----------
locator : `~matplotlib.ticker.Locator`

### Function: set_minor_locator(self, locator)

**Description:** Set the locator of the minor ticker.

Parameters
----------
locator : `~matplotlib.ticker.Locator`

### Function: set_pickradius(self, pickradius)

**Description:** Set the depth of the axis used by the picker.

Parameters
----------
pickradius : float
    The acceptance radius for containment tests.
    See also `.Axis.contains`.

### Function: _format_with_dict(tickd, x, pos)

### Function: set_ticklabels(self, labels)

**Description:** [*Discouraged*] Set this Axis' tick labels with list of string labels.

.. admonition:: Discouraged

    The use of this method is discouraged, because of the dependency on
    tick positions. In most cases, you'll want to use
    ``Axes.set_[x/y/z]ticks(positions, labels)`` or ``Axis.set_ticks``
    instead.

    If you are using this method, you should always fix the tick
    positions before, e.g. by using `.Axis.set_ticks` or by explicitly
    setting a `~.ticker.FixedLocator`. Otherwise, ticks are free to
    move and the labels may end up in unexpected positions.

Parameters
----------
labels : sequence of str or of `.Text`\s
    Texts for labeling each tick location in the sequence set by
    `.Axis.set_ticks`; the number of labels must match the number of locations.
    The labels are used as is, via a `.FixedFormatter` (without further
    formatting).

minor : bool
    If True, set minor ticks instead of major ticks.

fontdict : dict, optional

    .. admonition:: Discouraged

       The use of *fontdict* is discouraged. Parameters should be passed as
       individual keyword arguments or using dictionary-unpacking
       ``set_ticklabels(..., **fontdict)``.

    A dictionary controlling the appearance of the ticklabels.
    The default *fontdict* is::

       {'fontsize': rcParams['axes.titlesize'],
        'fontweight': rcParams['axes.titleweight'],
        'verticalalignment': 'baseline',
        'horizontalalignment': loc}

**kwargs
    Text properties.

    .. warning::

        This only sets the properties of the current ticks, which is
        only sufficient for static plots.

        Ticks are not guaranteed to be persistent. Various operations
        can create, delete and modify the Tick instances. There is an
        imminent risk that these settings can get lost if you work on
        the figure further (including also panning/zooming on a
        displayed figure).

        Use `.set_tick_params` instead if possible.

Returns
-------
list of `.Text`\s
    For each tick, includes ``tick.label1`` if it is visible, then
    ``tick.label2`` if it is visible, in that order.

### Function: _set_tick_locations(self, ticks)

### Function: set_ticks(self, ticks, labels)

**Description:** Set this Axis' tick locations and optionally tick labels.

If necessary, the view limits of the Axis are expanded so that all
given ticks are visible.

Parameters
----------
ticks : 1D array-like
    Array of tick locations (either floats or in axis units). The axis
    `.Locator` is replaced by a `~.ticker.FixedLocator`.

    Pass an empty list (``set_ticks([])``) to remove all ticks.

    Some tick formatters will not label arbitrary tick positions;
    e.g. log formatters only label decade ticks by default. In
    such a case you can set a formatter explicitly on the axis
    using `.Axis.set_major_formatter` or provide formatted
    *labels* yourself.

labels : list of str, optional
    Tick labels for each location in *ticks*; must have the same length as
    *ticks*. If set, the labels are used as is, via a `.FixedFormatter`.
    If not set, the labels are generated using the axis tick `.Formatter`.

minor : bool, default: False
    If ``False``, set only the major ticks; if ``True``, only the minor ticks.

**kwargs
    `.Text` properties for the labels. Using these is only allowed if
    you pass *labels*. In other cases, please use `~.Axes.tick_params`.

Notes
-----
The mandatory expansion of the view limits is an intentional design
choice to prevent the surprise of a non-visible tick. If you need
other limits, you should set the limits explicitly after setting the
ticks.

### Function: _get_tick_boxes_siblings(self, renderer)

**Description:** Get the bounding boxes for this `.axis` and its siblings
as set by `.Figure.align_xlabels` or  `.Figure.align_ylabels`.

By default, it just gets bboxes for *self*.

### Function: _update_label_position(self, renderer)

**Description:** Update the label position based on the bounding box enclosing
all the ticklabels and axis spine.

### Function: _update_offset_text_position(self, bboxes, bboxes2)

**Description:** Update the offset text position based on the sequence of bounding
boxes of all the ticklabels.

### Function: axis_date(self, tz)

**Description:** Set up axis ticks and labels to treat data along this Axis as dates.

Parameters
----------
tz : str or `datetime.tzinfo`, default: :rc:`timezone`
    The timezone used to create date labels.

### Function: get_tick_space(self)

**Description:** Return the estimated number of ticks that can fit on the axis.

### Function: _get_ticks_position(self)

**Description:** Helper for `XAxis.get_ticks_position` and `YAxis.get_ticks_position`.

Check the visibility of tick1line, label1, tick2line, and label2 on
the first major and the first minor ticks, and return

- 1 if only tick1line and label1 are visible (which corresponds to
  "bottom" for the x-axis and "left" for the y-axis);
- 2 if only tick2line and label2 are visible (which corresponds to
  "top" for the x-axis and "right" for the y-axis);
- "default" if only tick1line, tick2line and label1 are visible;
- "unknown" otherwise.

### Function: get_label_position(self)

**Description:** Return the label position (top or bottom)

### Function: set_label_position(self, position)

**Description:** Set the label position (top or bottom)

Parameters
----------
position : {'top', 'bottom'}

### Function: get_minpos(self)

### Function: getter(self)

### Function: setter(self, vmin, vmax, ignore)

### Function: __init__(self)

### Function: _init(self)

**Description:** Initialize the label and offsetText instance values and
`label_position` / `offset_text_position`.

### Function: contains(self, mouseevent)

**Description:** Test whether the mouse event occurred in the x-axis.

### Function: set_label_position(self, position)

**Description:** Set the label position (top or bottom)

Parameters
----------
position : {'top', 'bottom'}

### Function: _update_label_position(self, renderer)

**Description:** Update the label position based on the bounding box enclosing
all the ticklabels and axis spine

### Function: _update_offset_text_position(self, bboxes, bboxes2)

**Description:** Update the offset_text position based on the sequence of bounding
boxes of all the ticklabels

### Function: set_ticks_position(self, position)

**Description:** Set the ticks position.

Parameters
----------
position : {'top', 'bottom', 'both', 'default', 'none'}
    'both' sets the ticks to appear on both positions, but does not
    change the tick labels.  'default' resets the tick positions to
    the default: ticks on both positions, labels at bottom.  'none'
    can be used if you don't want any ticks. 'none' and 'both'
    affect only the ticks, not the labels.

### Function: tick_top(self)

**Description:** Move ticks and ticklabels (if present) to the top of the Axes.

### Function: tick_bottom(self)

**Description:** Move ticks and ticklabels (if present) to the bottom of the Axes.

### Function: get_ticks_position(self)

**Description:** Return the ticks position ("top", "bottom", "default", or "unknown").

### Function: get_minpos(self)

### Function: set_default_intervals(self)

### Function: get_tick_space(self)

### Function: __init__(self)

### Function: _init(self)

**Description:** Initialize the label and offsetText instance values and
`label_position` / `offset_text_position`.

### Function: contains(self, mouseevent)

### Function: set_label_position(self, position)

**Description:** Set the label position (left or right)

Parameters
----------
position : {'left', 'right'}

### Function: _update_label_position(self, renderer)

**Description:** Update the label position based on the bounding box enclosing
all the ticklabels and axis spine

### Function: _update_offset_text_position(self, bboxes, bboxes2)

**Description:** Update the offset_text position based on the sequence of bounding
boxes of all the ticklabels

### Function: set_offset_position(self, position)

**Description:** Parameters
----------
position : {'left', 'right'}

### Function: set_ticks_position(self, position)

**Description:** Set the ticks position.

Parameters
----------
position : {'left', 'right', 'both', 'default', 'none'}
    'both' sets the ticks to appear on both positions, but does not
    change the tick labels.  'default' resets the tick positions to
    the default: ticks on both positions, labels at left.  'none'
    can be used if you don't want any ticks. 'none' and 'both'
    affect only the ticks, not the labels.

### Function: tick_right(self)

**Description:** Move ticks and ticklabels (if present) to the right of the Axes.

### Function: tick_left(self)

**Description:** Move ticks and ticklabels (if present) to the left of the Axes.

### Function: get_ticks_position(self)

**Description:** Return the ticks position ("left", "right", "default", or "unknown").

### Function: get_minpos(self)

### Function: set_default_intervals(self)

### Function: get_tick_space(self)
