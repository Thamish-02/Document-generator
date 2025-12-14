## AI Summary

A file named polar.py.


### Function: _apply_theta_transforms_warn()

## Class: PolarTransform

**Description:** The base polar transform.

This transform maps polar coordinates :math:`\theta, r` into Cartesian
coordinates :math:`x, y = r \cos(\theta), r \sin(\theta)`
(but does not fully transform into Axes coordinates or
handle positioning in screen space).

This transformation is designed to be applied to data after any scaling
along the radial axis (e.g. log-scaling) has been applied to the input
data.

Path segments at a fixed radius are automatically transformed to circular
arcs as long as ``path._interpolation_steps > 1``.

## Class: PolarAffine

**Description:** The affine part of the polar projection.

Scales the output so that maximum radius rests on the edge of the Axes
circle and the origin is mapped to (0.5, 0.5). The transform applied is
the same to x and y components and given by:

.. math::

    x_{1} = 0.5 \left [ \frac{x_{0}}{(r_{\max} - r_{\min})} + 1 \right ]

:math:`r_{\min}, r_{\max}` are the minimum and maximum radial limits after
any scaling (e.g. log scaling) has been removed.

## Class: InvertedPolarTransform

**Description:** The inverse of the polar transform, mapping Cartesian
coordinate space *x* and *y* back to *theta* and *r*.

## Class: ThetaFormatter

**Description:** Used to format the *theta* tick labels.  Converts the native
unit of radians into degrees and adds a degree symbol.

## Class: _AxisWrapper

## Class: ThetaLocator

**Description:** Used to locate theta ticks.

This will work the same as the base locator except in the case that the
view spans the entire circle. In such cases, the previously used default
locations of every 45 degrees are returned.

## Class: ThetaTick

**Description:** A theta-axis tick.

This subclass of `.XTick` provides angular ticks with some small
modification to their re-positioning such that ticks are rotated based on
tick location. This results in ticks that are correctly perpendicular to
the arc spine.

When 'auto' rotation is enabled, labels are also rotated to be parallel to
the spine. The label padding is also applied here since it's not possible
to use a generic axes transform to produce tick-specific padding.

## Class: ThetaAxis

**Description:** A theta Axis.

This overrides certain properties of an `.XAxis` to provide special-casing
for an angular axis.

## Class: RadialLocator

**Description:** Used to locate radius ticks.

Ensures that all ticks are strictly positive.  For all other tasks, it
delegates to the base `.Locator` (which may be different depending on the
scale of the *r*-axis).

## Class: _ThetaShift

**Description:** Apply a padding shift based on axes theta limits.

This is used to create padding for radial ticks.

Parameters
----------
axes : `~matplotlib.axes.Axes`
    The owning Axes; used to determine limits.
pad : float
    The padding to apply, in points.
mode : {'min', 'max', 'rlabel'}
    Whether to shift away from the start (``'min'``) or the end (``'max'``)
    of the axes, or using the rlabel position (``'rlabel'``).

## Class: RadialTick

**Description:** A radial-axis tick.

This subclass of `.YTick` provides radial ticks with some small
modification to their re-positioning such that ticks are rotated based on
axes limits.  This results in ticks that are correctly perpendicular to
the spine. Labels are also rotated to be perpendicular to the spine, when
'auto' rotation is enabled.

## Class: RadialAxis

**Description:** A radial Axis.

This overrides certain properties of a `.YAxis` to provide special-casing
for a radial axis.

### Function: _is_full_circle_deg(thetamin, thetamax)

**Description:** Determine if a wedge (in degrees) spans the full circle.

The condition is derived from :class:`~matplotlib.patches.Wedge`.

### Function: _is_full_circle_rad(thetamin, thetamax)

**Description:** Determine if a wedge (in radians) spans the full circle.

The condition is derived from :class:`~matplotlib.patches.Wedge`.

## Class: _WedgeBbox

**Description:** Transform (theta, r) wedge Bbox into Axes bounding box.

Parameters
----------
center : (float, float)
    Center of the wedge
viewLim : `~matplotlib.transforms.Bbox`
    Bbox determining the boundaries of the wedge
originLim : `~matplotlib.transforms.Bbox`
    Bbox determining the origin for the wedge, if different from *viewLim*

## Class: PolarAxes

**Description:** A polar graph projection, where the input dimensions are *theta*, *r*.

Theta starts pointing east and goes anti-clockwise.

### Function: __init__(self, axis, use_rmin)

**Description:** Parameters
----------
axis : `~matplotlib.axis.Axis`, optional
    Axis associated with this transform. This is used to get the
    minimum radial limit.
use_rmin : `bool`, optional
    If ``True``, subtract the minimum radial axis limit before
    transforming to Cartesian coordinates. *axis* must also be
    specified for this to take effect.

### Function: _get_rorigin(self)

### Function: transform_non_affine(self, values)

### Function: transform_path_non_affine(self, path)

### Function: inverted(self)

### Function: __init__(self, scale_transform, limits)

**Description:** Parameters
----------
scale_transform : `~matplotlib.transforms.Transform`
    Scaling transform for the data. This is used to remove any scaling
    from the radial view limits.
limits : `~matplotlib.transforms.BboxBase`
    View limits of the data. The only part of its bounds that is used
    is the y limits (for the radius limits).

### Function: get_matrix(self)

### Function: __init__(self, axis, use_rmin)

**Description:** Parameters
----------
axis : `~matplotlib.axis.Axis`, optional
    Axis associated with this transform. This is used to get the
    minimum radial limit.
use_rmin : `bool`, optional
    If ``True``, add the minimum radial axis limit after
    transforming from Cartesian coordinates. *axis* must also be
    specified for this to take effect.

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __call__(self, x, pos)

### Function: __init__(self, axis)

### Function: get_view_interval(self)

### Function: set_view_interval(self, vmin, vmax)

### Function: get_minpos(self)

### Function: get_data_interval(self)

### Function: set_data_interval(self, vmin, vmax)

### Function: get_tick_space(self)

### Function: __init__(self, base)

### Function: set_axis(self, axis)

### Function: __call__(self)

### Function: view_limits(self, vmin, vmax)

### Function: __init__(self, axes)

### Function: _apply_params(self)

### Function: _update_padding(self, pad, angle)

### Function: update_position(self, loc)

### Function: _wrap_locator_formatter(self)

### Function: clear(self)

### Function: _set_scale(self, value)

### Function: _copy_tick_props(self, src, dest)

**Description:** Copy the props from src tick to dest tick.

### Function: __init__(self, base, axes)

### Function: set_axis(self, axis)

### Function: __call__(self)

### Function: _zero_in_bounds(self)

**Description:** Return True if zero is within the valid values for the
scale of the radial axis.

### Function: nonsingular(self, vmin, vmax)

### Function: view_limits(self, vmin, vmax)

### Function: __init__(self, axes, pad, mode)

### Function: get_matrix(self)

### Function: __init__(self)

### Function: _determine_anchor(self, mode, angle, start)

### Function: update_position(self, loc)

### Function: __init__(self)

### Function: _wrap_locator_formatter(self)

### Function: clear(self)

### Function: _set_scale(self, value)

### Function: __init__(self, center, viewLim, originLim)

### Function: get_points(self)

### Function: __init__(self)

### Function: clear(self)

### Function: _init_axis(self)

### Function: _set_lim_and_transforms(self)

### Function: get_xaxis_transform(self, which)

### Function: get_xaxis_text1_transform(self, pad)

### Function: get_xaxis_text2_transform(self, pad)

### Function: get_yaxis_transform(self, which)

### Function: get_yaxis_text1_transform(self, pad)

### Function: get_yaxis_text2_transform(self, pad)

### Function: draw(self, renderer)

### Function: _gen_axes_patch(self)

### Function: _gen_axes_spines(self)

### Function: set_thetamax(self, thetamax)

**Description:** Set the maximum theta limit in degrees.

### Function: get_thetamax(self)

**Description:** Return the maximum theta limit in degrees.

### Function: set_thetamin(self, thetamin)

**Description:** Set the minimum theta limit in degrees.

### Function: get_thetamin(self)

**Description:** Get the minimum theta limit in degrees.

### Function: set_thetalim(self)

**Description:** Set the minimum and maximum theta values.

Can take the following signatures:

- ``set_thetalim(minval, maxval)``: Set the limits in radians.
- ``set_thetalim(thetamin=minval, thetamax=maxval)``: Set the limits
  in degrees.

where minval and maxval are the minimum and maximum limits. Values are
wrapped in to the range :math:`[0, 2\pi]` (in radians), so for example
it is possible to do ``set_thetalim(-np.pi / 2, np.pi / 2)`` to have
an axis symmetric around 0. A ValueError is raised if the absolute
angle difference is larger than a full circle.

### Function: set_theta_offset(self, offset)

**Description:** Set the offset for the location of 0 in radians.

### Function: get_theta_offset(self)

**Description:** Get the offset for the location of 0 in radians.

### Function: set_theta_zero_location(self, loc, offset)

**Description:** Set the location of theta's zero.

This simply calls `set_theta_offset` with the correct value in radians.

Parameters
----------
loc : str
    May be one of "N", "NW", "W", "SW", "S", "SE", "E", or "NE".
offset : float, default: 0
    An offset in degrees to apply from the specified *loc*. **Note:**
    this offset is *always* applied counter-clockwise regardless of
    the direction setting.

### Function: set_theta_direction(self, direction)

**Description:** Set the direction in which theta increases.

clockwise, -1:
   Theta increases in the clockwise direction

counterclockwise, anticlockwise, 1:
   Theta increases in the counterclockwise direction

### Function: get_theta_direction(self)

**Description:** Get the direction in which theta increases.

-1:
   Theta increases in the clockwise direction

1:
   Theta increases in the counterclockwise direction

### Function: set_rmax(self, rmax)

**Description:** Set the outer radial limit.

Parameters
----------
rmax : float

### Function: get_rmax(self)

**Description:** Returns
-------
float
    Outer radial limit.

### Function: set_rmin(self, rmin)

**Description:** Set the inner radial limit.

Parameters
----------
rmin : float

### Function: get_rmin(self)

**Description:** Returns
-------
float
    The inner radial limit.

### Function: set_rorigin(self, rorigin)

**Description:** Update the radial origin.

Parameters
----------
rorigin : float

### Function: get_rorigin(self)

**Description:** Returns
-------
float

### Function: get_rsign(self)

### Function: set_rlim(self, bottom, top)

**Description:** Set the radial axis view limits.

This function behaves like `.Axes.set_ylim`, but additionally supports
*rmin* and *rmax* as aliases for *bottom* and *top*.

See Also
--------
.Axes.set_ylim

### Function: get_rlabel_position(self)

**Description:** Returns
-------
float
    The theta position of the radius labels in degrees.

### Function: set_rlabel_position(self, value)

**Description:** Update the theta position of the radius labels.

Parameters
----------
value : number
    The angular position of the radius labels in degrees.

### Function: set_yscale(self)

### Function: set_rscale(self)

### Function: set_rticks(self)

### Function: set_thetagrids(self, angles, labels, fmt)

**Description:** Set the theta gridlines in a polar plot.

Parameters
----------
angles : tuple with floats, degrees
    The angles of the theta gridlines.

labels : tuple with strings or None
    The labels to use at each theta gridline. The
    `.projections.polar.ThetaFormatter` will be used if None.

fmt : str or None
    Format string used in `matplotlib.ticker.FormatStrFormatter`.
    For example '%f'. Note that the angle that is used is in
    radians.

Returns
-------
lines : list of `.lines.Line2D`
    The theta gridlines.

labels : list of `.text.Text`
    The tick labels.

Other Parameters
----------------
**kwargs
    *kwargs* are optional `.Text` properties for the labels.

    .. warning::

        This only sets the properties of the current ticks.
        Ticks are not guaranteed to be persistent. Various operations
        can create, delete and modify the Tick instances. There is an
        imminent risk that these settings can get lost if you work on
        the figure further (including also panning/zooming on a
        displayed figure).

        Use `.set_tick_params` instead if possible.

See Also
--------
.PolarAxes.set_rgrids
.Axis.get_gridlines
.Axis.get_ticklabels

### Function: set_rgrids(self, radii, labels, angle, fmt)

**Description:** Set the radial gridlines on a polar plot.

Parameters
----------
radii : tuple with floats
    The radii for the radial gridlines

labels : tuple with strings or None
    The labels to use at each radial gridline. The
    `matplotlib.ticker.ScalarFormatter` will be used if None.

angle : float
    The angular position of the radius labels in degrees.

fmt : str or None
    Format string used in `matplotlib.ticker.FormatStrFormatter`.
    For example '%f'.

Returns
-------
lines : list of `.lines.Line2D`
    The radial gridlines.

labels : list of `.text.Text`
    The tick labels.

Other Parameters
----------------
**kwargs
    *kwargs* are optional `.Text` properties for the labels.

    .. warning::

        This only sets the properties of the current ticks.
        Ticks are not guaranteed to be persistent. Various operations
        can create, delete and modify the Tick instances. There is an
        imminent risk that these settings can get lost if you work on
        the figure further (including also panning/zooming on a
        displayed figure).

        Use `.set_tick_params` instead if possible.

See Also
--------
.PolarAxes.set_thetagrids
.Axis.get_gridlines
.Axis.get_ticklabels

### Function: format_coord(self, theta, r)

### Function: get_data_ratio(self)

**Description:** Return the aspect ratio of the data itself.  For a polar plot,
this should always be 1.0

### Function: can_zoom(self)

**Description:** Return whether this Axes supports the zoom box button functionality.

A polar Axes does not support zoom boxes.

### Function: can_pan(self)

**Description:** Return whether this Axes supports the pan/zoom button functionality.

For a polar Axes, this is slightly misleading. Both panning and
zooming are performed by the same button. Panning is performed
in azimuth while zooming is done along the radial.

### Function: start_pan(self, x, y, button)

### Function: end_pan(self)

### Function: drag_pan(self, button, key, x, y)

### Function: format_sig(value, delta, opt, fmt)
