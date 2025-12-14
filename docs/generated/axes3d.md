## AI Summary

A file named axes3d.py.


## Class: Axes3D

**Description:** 3D Axes object.

.. note::

    As a user, you do not instantiate Axes directly, but use Axes creation
    methods instead; e.g. from `.pyplot` or `.Figure`:
    `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` or `.Figure.add_axes`.

### Function: get_test_data(delta)

**Description:** Return a tuple X, Y, Z with a test data set.

## Class: _Quaternion

**Description:** Quaternions
consisting of scalar, along 1, and vector, with components along i, j, k

### Function: __init__(self, fig, rect)

**Description:** Parameters
----------
fig : Figure
    The parent figure.
rect : tuple (left, bottom, width, height), default: None.
    The ``(left, bottom, width, height)`` Axes position.
elev : float, default: 30
    The elevation angle in degrees rotates the camera above and below
    the x-y plane, with a positive angle corresponding to a location
    above the plane.
azim : float, default: -60
    The azimuthal angle in degrees rotates the camera about the z axis,
    with a positive angle corresponding to a right-handed rotation. In
    other words, a positive azimuth rotates the camera about the origin
    from its location along the +x axis towards the +y axis.
roll : float, default: 0
    The roll angle in degrees rotates the camera about the viewing
    axis. A positive angle spins the camera clockwise, causing the
    scene to rotate counter-clockwise.
shareview : Axes3D, optional
    Other Axes to share view angles with.  Note that it is not possible
    to unshare axes.
sharez : Axes3D, optional
    Other Axes to share z-limits with.  Note that it is not possible to
    unshare axes.
proj_type : {'persp', 'ortho'}
    The projection type, default 'persp'.
focal_length : float, default: None
    For a projection type of 'persp', the focal length of the virtual
    camera. Must be > 0. If None, defaults to 1.
    For a projection type of 'ortho', must be set to either None
    or infinity (numpy.inf). If None, defaults to infinity.
    The focal length can be computed from a desired Field Of View via
    the equation: focal_length = 1/tan(FOV/2)
box_aspect : 3-tuple of floats, default: None
    Changes the physical dimensions of the Axes3D, such that the ratio
    of the axis lengths in display units is x:y:z.
    If None, defaults to 4:4:3
computed_zorder : bool, default: True
    If True, the draw order is computed based on the average position
    of the `.Artist`\s along the view direction.
    Set to False if you want to manually control the order in which
    Artists are drawn on top of each other using their *zorder*
    attribute. This can be used for fine-tuning if the automatic order
    does not produce the desired result. Note however, that a manual
    zorder will only be correct for a limited view angle. If the figure
    is rotated by the user, it will look wrong from certain angles.

**kwargs
    Other optional keyword arguments:

    %(Axes3D:kwdoc)s

### Function: set_axis_off(self)

### Function: set_axis_on(self)

### Function: convert_zunits(self, z)

**Description:** For artists in an Axes, if the zaxis has units support,
convert *z* using zaxis unit type

### Function: set_top_view(self)

### Function: _init_axis(self)

**Description:** Init 3D Axes; overrides creation of regular X/Y Axes.

### Function: get_zaxis(self)

**Description:** Return the ``ZAxis`` (`~.axis3d.Axis`) instance.

### Function: _transformed_cube(self, vals)

**Description:** Return cube with limits from *vals* transformed by self.M.

### Function: set_aspect(self, aspect, adjustable, anchor, share)

**Description:** Set the aspect ratios.

Parameters
----------
aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
    Possible values:

    =========   ==================================================
    value       description
    =========   ==================================================
    'auto'      automatic; fill the position rectangle with data.
    'equal'     adapt all the axes to have equal aspect ratios.
    'equalxy'   adapt the x and y axes to have equal aspect ratios.
    'equalxz'   adapt the x and z axes to have equal aspect ratios.
    'equalyz'   adapt the y and z axes to have equal aspect ratios.
    =========   ==================================================

adjustable : None or {'box', 'datalim'}, optional
    If not *None*, this defines which parameter will be adjusted to
    meet the required aspect. See `.set_adjustable` for further
    details.

anchor : None or str or 2-tuple of float, optional
    If not *None*, this defines where the Axes will be drawn if there
    is extra space due to aspect constraints. The most common way to
    specify the anchor are abbreviations of cardinal directions:

    =====   =====================
    value   description
    =====   =====================
    'C'     centered
    'SW'    lower left corner
    'S'     middle of bottom edge
    'SE'    lower right corner
    etc.
    =====   =====================

    See `~.Axes.set_anchor` for further details.

share : bool, default: False
    If ``True``, apply the settings to all shared Axes.

See Also
--------
mpl_toolkits.mplot3d.axes3d.Axes3D.set_box_aspect

### Function: _equal_aspect_axis_indices(self, aspect)

**Description:** Get the indices for which of the x, y, z axes are constrained to have
equal aspect ratios.

Parameters
----------
aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
    See descriptions in docstring for `.set_aspect()`.

### Function: set_box_aspect(self, aspect)

**Description:** Set the Axes box aspect.

The box aspect is the ratio of height to width in display
units for each face of the box when viewed perpendicular to
that face.  This is not to be confused with the data aspect (see
`~.Axes3D.set_aspect`). The default ratios are 4:4:3 (x:y:z).

To simulate having equal aspect in data space, set the box
aspect to match your data range in each dimension.

*zoom* controls the overall size of the Axes3D in the figure.

Parameters
----------
aspect : 3-tuple of floats or None
    Changes the physical dimensions of the Axes3D, such that the ratio
    of the axis lengths in display units is x:y:z.
    If None, defaults to (4, 4, 3).

zoom : float, default: 1
    Control overall size of the Axes3D in the figure. Must be > 0.

### Function: apply_aspect(self, position)

### Function: draw(self, renderer)

### Function: get_axis_position(self)

### Function: update_datalim(self, xys)

**Description:** Not implemented in `~mpl_toolkits.mplot3d.axes3d.Axes3D`.

### Function: get_zmargin(self)

**Description:** Retrieve autoscaling margin of the z-axis.

.. versionadded:: 3.9

Returns
-------
zmargin : float

See Also
--------
mpl_toolkits.mplot3d.axes3d.Axes3D.set_zmargin

### Function: set_zmargin(self, m)

**Description:** Set padding of Z data limits prior to autoscaling.

*m* times the data interval will be added to each end of that interval
before it is used in autoscaling.  If *m* is negative, this will clip
the data range instead of expanding it.

For example, if your data is in the range [0, 2], a margin of 0.1 will
result in a range [-0.2, 2.2]; a margin of -0.1 will result in a range
of [0.2, 1.8].

Parameters
----------
m : float greater than -0.5

### Function: margins(self)

**Description:** Set or retrieve autoscaling margins.

See `.Axes.margins` for full documentation.  Because this function
applies to 3D Axes, it also takes a *z* argument, and returns
``(xmargin, ymargin, zmargin)``.

### Function: autoscale(self, enable, axis, tight)

**Description:** Convenience method for simple axis view autoscaling.

See `.Axes.autoscale` for full documentation.  Because this function
applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*
to 'both' autoscales all three axes.

### Function: auto_scale_xyz(self, X, Y, Z, had_data)

### Function: autoscale_view(self, tight, scalex, scaley, scalez)

**Description:** Autoscale the view limits using the data limits.

See `.Axes.autoscale_view` for full documentation.  Because this
function applies to 3D Axes, it also takes a *scalez* argument.

### Function: get_w_lims(self)

**Description:** Get 3D world limits.

### Function: _set_bound3d(self, get_bound, set_lim, axis_inverted, lower, upper, view_margin)

**Description:** Set 3D axis bounds.

### Function: set_xbound(self, lower, upper, view_margin)

**Description:** Set the lower and upper numerical bounds of the x-axis.

This method will honor axis inversion regardless of parameter order.
It will not change the autoscaling setting (`.get_autoscalex_on()`).

Parameters
----------
lower, upper : float or None
    The lower and upper bounds. If *None*, the respective axis bound
    is not modified.
view_margin : float or None
    The margin to apply to the bounds. If *None*, the margin is handled
    by `.set_xlim`.

See Also
--------
get_xbound
get_xlim, set_xlim
invert_xaxis, xaxis_inverted

### Function: set_ybound(self, lower, upper, view_margin)

**Description:** Set the lower and upper numerical bounds of the y-axis.

This method will honor axis inversion regardless of parameter order.
It will not change the autoscaling setting (`.get_autoscaley_on()`).

Parameters
----------
lower, upper : float or None
    The lower and upper bounds. If *None*, the respective axis bound
    is not modified.
view_margin : float or None
    The margin to apply to the bounds. If *None*, the margin is handled
    by `.set_ylim`.

See Also
--------
get_ybound
get_ylim, set_ylim
invert_yaxis, yaxis_inverted

### Function: set_zbound(self, lower, upper, view_margin)

**Description:** Set the lower and upper numerical bounds of the z-axis.
This method will honor axis inversion regardless of parameter order.
It will not change the autoscaling setting (`.get_autoscaley_on()`).

Parameters
----------
lower, upper : float or None
    The lower and upper bounds. If *None*, the respective axis bound
    is not modified.
view_margin : float or None
    The margin to apply to the bounds. If *None*, the margin is handled
    by `.set_zlim`.

See Also
--------
get_zbound
get_zlim, set_zlim
invert_zaxis, zaxis_inverted

### Function: _set_lim3d(self, axis, lower, upper)

**Description:** Set 3D axis limits.

### Function: set_xlim(self, left, right)

**Description:** Set the 3D x-axis view limits.

Parameters
----------
left : float, optional
    The left xlim in data coordinates. Passing *None* leaves the
    limit unchanged.

    The left and right xlims may also be passed as the tuple
    (*left*, *right*) as the first positional argument (or as
    the *left* keyword argument).

    .. ACCEPTS: (left: float, right: float)

right : float, optional
    The right xlim in data coordinates. Passing *None* leaves the
    limit unchanged.

emit : bool, default: True
    Whether to notify observers of limit change.

auto : bool or None, default: False
    Whether to turn on autoscaling of the x-axis. *True* turns on,
    *False* turns off, *None* leaves unchanged.

view_margin : float, optional
    The additional margin to apply to the limits.

xmin, xmax : float, optional
    They are equivalent to left and right respectively, and it is an
    error to pass both *xmin* and *left* or *xmax* and *right*.

Returns
-------
left, right : (float, float)
    The new x-axis limits in data coordinates.

See Also
--------
get_xlim
set_xbound, get_xbound
invert_xaxis, xaxis_inverted

Notes
-----
The *left* value may be greater than the *right* value, in which
case the x-axis values will decrease from *left* to *right*.

Examples
--------
>>> set_xlim(left, right)
>>> set_xlim((left, right))
>>> left, right = set_xlim(left, right)

One limit may be left unchanged.

>>> set_xlim(right=right_lim)

Limits may be passed in reverse order to flip the direction of
the x-axis. For example, suppose ``x`` represents depth of the
ocean in m. The x-axis limits might be set like the following
so 5000 m depth is at the left of the plot and the surface,
0 m, is at the right.

>>> set_xlim(5000, 0)

### Function: set_ylim(self, bottom, top)

**Description:** Set the 3D y-axis view limits.

Parameters
----------
bottom : float, optional
    The bottom ylim in data coordinates. Passing *None* leaves the
    limit unchanged.

    The bottom and top ylims may also be passed as the tuple
    (*bottom*, *top*) as the first positional argument (or as
    the *bottom* keyword argument).

    .. ACCEPTS: (bottom: float, top: float)

top : float, optional
    The top ylim in data coordinates. Passing *None* leaves the
    limit unchanged.

emit : bool, default: True
    Whether to notify observers of limit change.

auto : bool or None, default: False
    Whether to turn on autoscaling of the y-axis. *True* turns on,
    *False* turns off, *None* leaves unchanged.

view_margin : float, optional
    The additional margin to apply to the limits.

ymin, ymax : float, optional
    They are equivalent to bottom and top respectively, and it is an
    error to pass both *ymin* and *bottom* or *ymax* and *top*.

Returns
-------
bottom, top : (float, float)
    The new y-axis limits in data coordinates.

See Also
--------
get_ylim
set_ybound, get_ybound
invert_yaxis, yaxis_inverted

Notes
-----
The *bottom* value may be greater than the *top* value, in which
case the y-axis values will decrease from *bottom* to *top*.

Examples
--------
>>> set_ylim(bottom, top)
>>> set_ylim((bottom, top))
>>> bottom, top = set_ylim(bottom, top)

One limit may be left unchanged.

>>> set_ylim(top=top_lim)

Limits may be passed in reverse order to flip the direction of
the y-axis. For example, suppose ``y`` represents depth of the
ocean in m. The y-axis limits might be set like the following
so 5000 m depth is at the bottom of the plot and the surface,
0 m, is at the top.

>>> set_ylim(5000, 0)

### Function: set_zlim(self, bottom, top)

**Description:** Set the 3D z-axis view limits.

Parameters
----------
bottom : float, optional
    The bottom zlim in data coordinates. Passing *None* leaves the
    limit unchanged.

    The bottom and top zlims may also be passed as the tuple
    (*bottom*, *top*) as the first positional argument (or as
    the *bottom* keyword argument).

    .. ACCEPTS: (bottom: float, top: float)

top : float, optional
    The top zlim in data coordinates. Passing *None* leaves the
    limit unchanged.

emit : bool, default: True
    Whether to notify observers of limit change.

auto : bool or None, default: False
    Whether to turn on autoscaling of the z-axis. *True* turns on,
    *False* turns off, *None* leaves unchanged.

view_margin : float, optional
    The additional margin to apply to the limits.

zmin, zmax : float, optional
    They are equivalent to bottom and top respectively, and it is an
    error to pass both *zmin* and *bottom* or *zmax* and *top*.

Returns
-------
bottom, top : (float, float)
    The new z-axis limits in data coordinates.

See Also
--------
get_zlim
set_zbound, get_zbound
invert_zaxis, zaxis_inverted

Notes
-----
The *bottom* value may be greater than the *top* value, in which
case the z-axis values will decrease from *bottom* to *top*.

Examples
--------
>>> set_zlim(bottom, top)
>>> set_zlim((bottom, top))
>>> bottom, top = set_zlim(bottom, top)

One limit may be left unchanged.

>>> set_zlim(top=top_lim)

Limits may be passed in reverse order to flip the direction of
the z-axis. For example, suppose ``z`` represents depth of the
ocean in m. The z-axis limits might be set like the following
so 5000 m depth is at the bottom of the plot and the surface,
0 m, is at the top.

>>> set_zlim(5000, 0)

### Function: get_xlim(self)

### Function: get_ylim(self)

### Function: get_zlim(self)

**Description:** Return the 3D z-axis view limits.

Returns
-------
left, right : (float, float)
    The current z-axis limits in data coordinates.

See Also
--------
set_zlim
set_zbound, get_zbound
invert_zaxis, zaxis_inverted

Notes
-----
The z-axis may be inverted, in which case the *left* value will
be greater than the *right* value.

### Function: clabel(self)

**Description:** Currently not implemented for 3D Axes, and returns *None*.

### Function: view_init(self, elev, azim, roll, vertical_axis, share)

**Description:** Set the elevation and azimuth of the Axes in degrees (not radians).

This can be used to rotate the Axes programmatically.

To look normal to the primary planes, the following elevation and
azimuth angles can be used. A roll angle of 0, 90, 180, or 270 deg
will rotate these views while keeping the axes at right angles.

==========   ====  ====
view plane   elev  azim
==========   ====  ====
XY           90    -90
XZ           0     -90
YZ           0     0
-XY          -90   90
-XZ          0     90
-YZ          0     180
==========   ====  ====

Parameters
----------
elev : float, default: None
    The elevation angle in degrees rotates the camera above the plane
    pierced by the vertical axis, with a positive angle corresponding
    to a location above that plane. For example, with the default
    vertical axis of 'z', the elevation defines the angle of the camera
    location above the x-y plane.
    If None, then the initial value as specified in the `Axes3D`
    constructor is used.
azim : float, default: None
    The azimuthal angle in degrees rotates the camera about the
    vertical axis, with a positive angle corresponding to a
    right-handed rotation. For example, with the default vertical axis
    of 'z', a positive azimuth rotates the camera about the origin from
    its location along the +x axis towards the +y axis.
    If None, then the initial value as specified in the `Axes3D`
    constructor is used.
roll : float, default: None
    The roll angle in degrees rotates the camera about the viewing
    axis. A positive angle spins the camera clockwise, causing the
    scene to rotate counter-clockwise.
    If None, then the initial value as specified in the `Axes3D`
    constructor is used.
vertical_axis : {"z", "x", "y"}, default: "z"
    The axis to align vertically. *azim* rotates about this axis.
share : bool, default: False
    If ``True``, apply the settings to all Axes with shared views.

### Function: set_proj_type(self, proj_type, focal_length)

**Description:** Set the projection type.

Parameters
----------
proj_type : {'persp', 'ortho'}
    The projection type.
focal_length : float, default: None
    For a projection type of 'persp', the focal length of the virtual
    camera. Must be > 0. If None, defaults to 1.
    The focal length can be computed from a desired Field Of View via
    the equation: focal_length = 1/tan(FOV/2)

### Function: _roll_to_vertical(self, arr, reverse)

**Description:** Roll arrays to match the different vertical axis.

Parameters
----------
arr : ArrayLike
    Array to roll.
reverse : bool, default: False
    Reverse the direction of the roll.

### Function: get_proj(self)

**Description:** Create the projection matrix from the current viewing position.

### Function: mouse_init(self, rotate_btn, pan_btn, zoom_btn)

**Description:** Set the mouse buttons for 3D rotation and zooming.

Parameters
----------
rotate_btn : int or list of int, default: 1
    The mouse button or buttons to use for 3D rotation of the Axes.
pan_btn : int or list of int, default: 2
    The mouse button or buttons to use to pan the 3D Axes.
zoom_btn : int or list of int, default: 3
    The mouse button or buttons to use to zoom the 3D Axes.

### Function: disable_mouse_rotation(self)

**Description:** Disable mouse buttons for 3D rotation, panning, and zooming.

### Function: can_zoom(self)

### Function: can_pan(self)

### Function: sharez(self, other)

**Description:** Share the z-axis with *other*.

This is equivalent to passing ``sharez=other`` when constructing the
Axes, and cannot be used if the z-axis is already being shared with
another Axes.  Note that it is not possible to unshare axes.

### Function: shareview(self, other)

**Description:** Share the view angles with *other*.

This is equivalent to passing ``shareview=other`` when constructing the
Axes, and cannot be used if the view angles are already being shared
with another Axes.  Note that it is not possible to unshare axes.

### Function: clear(self)

### Function: _button_press(self, event)

### Function: _button_release(self, event)

### Function: _get_view(self)

### Function: _set_view(self, view)

### Function: format_zdata(self, z)

**Description:** Return *z* string formatted.  This function will use the
:attr:`fmt_zdata` attribute if it is callable, else will fall
back on the zaxis major formatter

### Function: format_coord(self, xv, yv, renderer)

**Description:** Return a string giving the current view rotation angles, or the x, y, z
coordinates of the point on the nearest axis pane underneath the mouse
cursor, depending on the mouse button pressed.

### Function: _rotation_coords(self)

**Description:** Return the rotation angles as a string.

### Function: _location_coords(self, xv, yv, renderer)

**Description:** Return the location on the axis pane underneath the cursor as a string.

### Function: _get_camera_loc(self)

**Description:** Returns the current camera location in data coordinates.

### Function: _calc_coord(self, xv, yv, renderer)

**Description:** Given the 2D view coordinates, find the point on the nearest axis pane
that lies directly below those coordinates. Returns a 3D point in data
coordinates.

### Function: _arcball(self, x, y)

**Description:** Convert a point (x, y) to a point on a virtual trackball.

This is Ken Shoemake's arcball (a sphere), modified
to soften the abrupt edge (optionally).
See: Ken Shoemake, "ARCBALL: A user interface for specifying
three-dimensional rotation using a mouse." in
Proceedings of Graphics Interface '92, 1992, pp. 151-156,
https://doi.org/10.20380/GI1992.18
The smoothing of the edge is inspired by Gavin Bell's arcball
(a sphere combined with a hyperbola), but here, the sphere
is combined with a section of a cylinder, so it has finite support.

### Function: _on_move(self, event)

**Description:** Mouse moving.

By default, button-1 rotates, button-2 pans, and button-3 zooms;
these buttons can be modified via `mouse_init`.

### Function: drag_pan(self, button, key, x, y)

### Function: _calc_view_axes(self, eye)

**Description:** Get the unit vectors for the viewing axes in data coordinates.
`u` is towards the right of the screen
`v` is towards the top of the screen
`w` is out of the screen

### Function: _set_view_from_bbox(self, bbox, direction, mode, twinx, twiny)

**Description:** Zoom in or out of the bounding box.

Will center the view in the center of the bounding box, and zoom by
the ratio of the size of the bounding box to the size of the Axes3D.

### Function: _zoom_data_limits(self, scale_u, scale_v, scale_w)

**Description:** Zoom in or out of a 3D plot.

Will scale the data limits by the scale factors. These will be
transformed to the x, y, z data axes based on the current view angles.
A scale factor > 1 zooms out and a scale factor < 1 zooms in.

For an Axes that has had its aspect ratio set to 'equal', 'equalxy',
'equalyz', or 'equalxz', the relevant axes are constrained to zoom
equally.

Parameters
----------
scale_u : float
    Scale factor for the u view axis (view screen horizontal).
scale_v : float
    Scale factor for the v view axis (view screen vertical).
scale_w : float
    Scale factor for the w view axis (view screen depth).

### Function: _scale_axis_limits(self, scale_x, scale_y, scale_z)

**Description:** Keeping the center of the x, y, and z data axes fixed, scale their
limits by scale factors. A scale factor > 1 zooms out and a scale
factor < 1 zooms in.

Parameters
----------
scale_x : float
    Scale factor for the x data axis.
scale_y : float
    Scale factor for the y data axis.
scale_z : float
    Scale factor for the z data axis.

### Function: _get_w_centers_ranges(self)

**Description:** Get 3D world centers and axis ranges.

### Function: set_zlabel(self, zlabel, fontdict, labelpad)

**Description:** Set zlabel.  See doc for `.set_ylabel` for description.

### Function: get_zlabel(self)

**Description:** Get the z-label text string.

### Function: grid(self, visible)

**Description:** Set / unset 3D grid.

.. note::

    Currently, this function does not behave the same as
    `.axes.Axes.grid`, but it is intended to eventually support that
    behavior.

### Function: tick_params(self, axis)

**Description:** Convenience method for changing the appearance of ticks and
tick labels.

See `.Axes.tick_params` for full documentation.  Because this function
applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*
to 'both' autoscales all three axes.

Also, because of how Axes3D objects are drawn very differently
from regular 2D Axes, some of these settings may have
ambiguous meaning.  For simplicity, the 'z' axis will
accept settings as if it was like the 'y' axis.

.. note::
   Axes3D currently ignores some of these settings.

### Function: invert_zaxis(self)

**Description:** Invert the z-axis.

See Also
--------
zaxis_inverted
get_zlim, set_zlim
get_zbound, set_zbound

### Function: get_zbound(self)

**Description:** Return the lower and upper z-axis bounds, in increasing order.

See Also
--------
set_zbound
get_zlim, set_zlim
invert_zaxis, zaxis_inverted

### Function: text(self, x, y, z, s, zdir)

**Description:** Add the text *s* to the 3D Axes at location *x*, *y*, *z* in data coordinates.

Parameters
----------
x, y, z : float
    The position to place the text.
s : str
    The text.
zdir : {'x', 'y', 'z', 3-tuple}, optional
    The direction to be used as the z-direction. Default: 'z'.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide text that is outside the axes view limits.

    .. versionadded:: 3.10
**kwargs
    Other arguments are forwarded to `matplotlib.axes.Axes.text`.

Returns
-------
`.Text3D`
    The created `.Text3D` instance.

### Function: plot(self, xs, ys)

**Description:** Plot 2D or 3D data.

Parameters
----------
xs : 1D array-like
    x coordinates of vertices.
ys : 1D array-like
    y coordinates of vertices.
zs : float or 1D array-like
    z coordinates of vertices; either one for all points or one for
    each point.
zdir : {'x', 'y', 'z'}, default: 'z'
    When plotting 2D data, the direction to use as z.
axlim_clip : bool, default: False
    Whether to hide data that is outside the axes view limits.

    .. versionadded:: 3.10
**kwargs
    Other arguments are forwarded to `matplotlib.axes.Axes.plot`.

### Function: fill_between(self, x1, y1, z1, x2, y2, z2)

**Description:** Fill the area between two 3D curves.

The curves are defined by the points (*x1*, *y1*, *z1*) and
(*x2*, *y2*, *z2*). This creates one or multiple quadrangle
polygons that are filled. All points must be the same length N, or a
single value to be used for all points.

Parameters
----------
x1, y1, z1 : float or 1D array-like
    x, y, and z  coordinates of vertices for 1st line.

x2, y2, z2 : float or 1D array-like
    x, y, and z coordinates of vertices for 2nd line.

where : array of bool (length N), optional
    Define *where* to exclude some regions from being filled. The
    filled regions are defined by the coordinates ``pts[where]``,
    for all x, y, and z pts. More precisely, fill between ``pts[i]``
    and ``pts[i+1]`` if ``where[i] and where[i+1]``. Note that this
    definition implies that an isolated *True* value between two
    *False* values in *where* will not result in filling. Both sides of
    the *True* position remain unfilled due to the adjacent *False*
    values.

mode : {'quad', 'polygon', 'auto'}, default: 'auto'
    The fill mode. One of:

    - 'quad':  A separate quadrilateral polygon is created for each
      pair of subsequent points in the two lines.
    - 'polygon': The two lines are connected to form a single polygon.
      This is faster and can render more cleanly for simple shapes
      (e.g. for filling between two lines that lie within a plane).
    - 'auto': If the points all lie on the same 3D plane, 'polygon' is
      used. Otherwise, 'quad' is used.

facecolors : list of :mpltype:`color`, default: None
    Colors of each individual patch, or a single color to be used for
    all patches.

shade : bool, default: None
    Whether to shade the facecolors. If *None*, then defaults to *True*
    for 'quad' mode and *False* for 'polygon' mode.

axlim_clip : bool, default: False
    Whether to hide data that is outside the axes view limits.

    .. versionadded:: 3.10

**kwargs
    All other keyword arguments are passed on to `.Poly3DCollection`.

Returns
-------
`.Poly3DCollection`
    A `.Poly3DCollection` containing the plotted polygons.

### Function: plot_surface(self, X, Y, Z)

**Description:** Create a surface plot.

By default, it will be colored in shades of a solid color, but it also
supports colormapping by supplying the *cmap* argument.

.. note::

   The *rcount* and *ccount* kwargs, which both default to 50,
   determine the maximum number of samples used in each direction.  If
   the input data is larger, it will be downsampled (by slicing) to
   these numbers of points.

.. note::

   To maximize rendering speed consider setting *rstride* and *cstride*
   to divisors of the number of rows minus 1 and columns minus 1
   respectively. For example, given 51 rows rstride can be any of the
   divisors of 50.

   Similarly, a setting of *rstride* and *cstride* equal to 1 (or
   *rcount* and *ccount* equal the number of rows and columns) can use
   the optimized path.

Parameters
----------
X, Y, Z : 2D arrays
    Data values.

rcount, ccount : int
    Maximum number of samples used in each direction.  If the input
    data is larger, it will be downsampled (by slicing) to these
    numbers of points.  Defaults to 50.

rstride, cstride : int
    Downsampling stride in each direction.  These arguments are
    mutually exclusive with *rcount* and *ccount*.  If only one of
    *rstride* or *cstride* is set, the other defaults to 10.

    'classic' mode uses a default of ``rstride = cstride = 10`` instead
    of the new default of ``rcount = ccount = 50``.

color : :mpltype:`color`
    Color of the surface patches.

cmap : Colormap, optional
    Colormap of the surface patches.

facecolors : list of :mpltype:`color`
    Colors of each individual patch.

norm : `~matplotlib.colors.Normalize`, optional
    Normalization for the colormap.

vmin, vmax : float, optional
    Bounds for the normalization.

shade : bool, default: True
    Whether to shade the facecolors.  Shading is always disabled when
    *cmap* is specified.

lightsource : `~matplotlib.colors.LightSource`, optional
    The lightsource to use when *shade* is True.

axlim_clip : bool, default: False
    Whether to hide patches with a vertex outside the axes view limits.

    .. versionadded:: 3.10

**kwargs
    Other keyword arguments are forwarded to `.Poly3DCollection`.

### Function: plot_wireframe(self, X, Y, Z)

**Description:** Plot a 3D wireframe.

.. note::

   The *rcount* and *ccount* kwargs, which both default to 50,
   determine the maximum number of samples used in each direction.  If
   the input data is larger, it will be downsampled (by slicing) to
   these numbers of points.

Parameters
----------
X, Y, Z : 2D arrays
    Data values.

axlim_clip : bool, default: False
    Whether to hide lines and patches with vertices outside the axes
    view limits.

    .. versionadded:: 3.10

rcount, ccount : int
    Maximum number of samples used in each direction.  If the input
    data is larger, it will be downsampled (by slicing) to these
    numbers of points.  Setting a count to zero causes the data to be
    not sampled in the corresponding direction, producing a 3D line
    plot rather than a wireframe plot.  Defaults to 50.

rstride, cstride : int
    Downsampling stride in each direction.  These arguments are
    mutually exclusive with *rcount* and *ccount*.  If only one of
    *rstride* or *cstride* is set, the other defaults to 1.  Setting a
    stride to zero causes the data to be not sampled in the
    corresponding direction, producing a 3D line plot rather than a
    wireframe plot.

    'classic' mode uses a default of ``rstride = cstride = 1`` instead
    of the new default of ``rcount = ccount = 50``.

**kwargs
    Other keyword arguments are forwarded to `.Line3DCollection`.

### Function: plot_trisurf(self)

**Description:** Plot a triangulated surface.

The (optional) triangulation can be specified in one of two ways;
either::

  plot_trisurf(triangulation, ...)

where triangulation is a `~matplotlib.tri.Triangulation` object, or::

  plot_trisurf(X, Y, ...)
  plot_trisurf(X, Y, triangles, ...)
  plot_trisurf(X, Y, triangles=triangles, ...)

in which case a Triangulation object will be created.  See
`.Triangulation` for an explanation of these possibilities.

The remaining arguments are::

  plot_trisurf(..., Z)

where *Z* is the array of values to contour, one per point
in the triangulation.

Parameters
----------
X, Y, Z : array-like
    Data values as 1D arrays.
color
    Color of the surface patches.
cmap
    A colormap for the surface patches.
norm : `~matplotlib.colors.Normalize`, optional
    An instance of Normalize to map values to colors.
vmin, vmax : float, optional
    Minimum and maximum value to map.
shade : bool, default: True
    Whether to shade the facecolors.  Shading is always disabled when
    *cmap* is specified.
lightsource : `~matplotlib.colors.LightSource`, optional
    The lightsource to use when *shade* is True.
axlim_clip : bool, default: False
    Whether to hide patches with a vertex outside the axes view limits.

    .. versionadded:: 3.10
**kwargs
    All other keyword arguments are passed on to
    :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`

Examples
--------
.. plot:: gallery/mplot3d/trisurf3d.py
.. plot:: gallery/mplot3d/trisurf3d_2.py

### Function: _3d_extend_contour(self, cset, stride)

**Description:** Extend a contour in 3D by creating

### Function: add_contour_set(self, cset, extend3d, stride, zdir, offset, axlim_clip)

### Function: add_contourf_set(self, cset, zdir, offset)

### Function: _add_contourf_set(self, cset, zdir, offset, axlim_clip)

**Description:** Returns
-------
levels : `numpy.ndarray`
    Levels at which the filled contours are added.

### Function: contour(self, X, Y, Z)

**Description:** Create a 3D contour plot.

Parameters
----------
X, Y, Z : array-like,
    Input data. See `.Axes.contour` for supported data shapes.
extend3d : bool, default: False
    Whether to extend contour in 3D.
stride : int, default: 5
    Step size for extending contour.
zdir : {'x', 'y', 'z'}, default: 'z'
    The direction to use.
offset : float, optional
    If specified, plot a projection of the contour lines at this
    position in a plane normal to *zdir*.
axlim_clip : bool, default: False
    Whether to hide lines with a vertex outside the axes view limits.

    .. versionadded:: 3.10
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

*args, **kwargs
    Other arguments are forwarded to `matplotlib.axes.Axes.contour`.

Returns
-------
matplotlib.contour.QuadContourSet

### Function: tricontour(self)

**Description:** Create a 3D contour plot.

.. note::
    This method currently produces incorrect output due to a
    longstanding bug in 3D PolyCollection rendering.

Parameters
----------
X, Y, Z : array-like
    Input data. See `.Axes.tricontour` for supported data shapes.
extend3d : bool, default: False
    Whether to extend contour in 3D.
stride : int, default: 5
    Step size for extending contour.
zdir : {'x', 'y', 'z'}, default: 'z'
    The direction to use.
offset : float, optional
    If specified, plot a projection of the contour lines at this
    position in a plane normal to *zdir*.
axlim_clip : bool, default: False
    Whether to hide lines with a vertex outside the axes view limits.

    .. versionadded:: 3.10
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER
*args, **kwargs
    Other arguments are forwarded to `matplotlib.axes.Axes.tricontour`.

Returns
-------
matplotlib.tri._tricontour.TriContourSet

### Function: _auto_scale_contourf(self, X, Y, Z, zdir, levels, had_data)

### Function: contourf(self, X, Y, Z)

**Description:** Create a 3D filled contour plot.

Parameters
----------
X, Y, Z : array-like
    Input data. See `.Axes.contourf` for supported data shapes.
zdir : {'x', 'y', 'z'}, default: 'z'
    The direction to use.
offset : float, optional
    If specified, plot a projection of the contour lines at this
    position in a plane normal to *zdir*.
axlim_clip : bool, default: False
    Whether to hide lines with a vertex outside the axes view limits.

    .. versionadded:: 3.10
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER
*args, **kwargs
    Other arguments are forwarded to `matplotlib.axes.Axes.contourf`.

Returns
-------
matplotlib.contour.QuadContourSet

### Function: tricontourf(self)

**Description:** Create a 3D filled contour plot.

.. note::
    This method currently produces incorrect output due to a
    longstanding bug in 3D PolyCollection rendering.

Parameters
----------
X, Y, Z : array-like
    Input data. See `.Axes.tricontourf` for supported data shapes.
zdir : {'x', 'y', 'z'}, default: 'z'
    The direction to use.
offset : float, optional
    If specified, plot a projection of the contour lines at this
    position in a plane normal to zdir.
axlim_clip : bool, default: False
    Whether to hide lines with a vertex outside the axes view limits.

    .. versionadded:: 3.10
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER
*args, **kwargs
    Other arguments are forwarded to
    `matplotlib.axes.Axes.tricontourf`.

Returns
-------
matplotlib.tri._tricontour.TriContourSet

### Function: add_collection3d(self, col, zs, zdir, autolim)

**Description:** Add a 3D collection object to the plot.

2D collection types are converted to a 3D version by
modifying the object and adding z coordinate information,
*zs* and *zdir*.

Supported 2D collection types are:

- `.PolyCollection`
- `.LineCollection`
- `.PatchCollection` (currently not supporting *autolim*)

Parameters
----------
col : `.Collection`
    A 2D collection object.
zs : float or array-like, default: 0
    The z-positions to be used for the 2D objects.
zdir : {'x', 'y', 'z'}, default: 'z'
    The direction to use for the z-positions.
autolim : bool, default: True
    Whether to update the data limits.
axlim_clip : bool, default: False
    Whether to hide the scatter points outside the axes view limits.

    .. versionadded:: 3.10

### Function: scatter(self, xs, ys, zs, zdir, s, c, depthshade)

**Description:** Create a scatter plot.

Parameters
----------
xs, ys : array-like
    The data positions.
zs : float or array-like, default: 0
    The z-positions. Either an array of the same length as *xs* and
    *ys* or a single value to place all points in the same plane.
zdir : {'x', 'y', 'z', '-x', '-y', '-z'}, default: 'z'
    The axis direction for the *zs*. This is useful when plotting 2D
    data on a 3D Axes. The data must be passed as *xs*, *ys*. Setting
    *zdir* to 'y' then plots the data to the x-z-plane.

    See also :doc:`/gallery/mplot3d/2dcollections3d`.

s : float or array-like, default: 20
    The marker size in points**2. Either an array of the same length
    as *xs* and *ys* or a single value to make all markers the same
    size.
c : :mpltype:`color`, sequence, or sequence of colors, optional
    The marker color. Possible values:

    - A single color format string.
    - A sequence of colors of length n.
    - A sequence of n numbers to be mapped to colors using *cmap* and
      *norm*.
    - A 2D array in which the rows are RGB or RGBA.

    For more details see the *c* argument of `~.axes.Axes.scatter`.
depthshade : bool, default: True
    Whether to shade the scatter markers to give the appearance of
    depth. Each call to ``scatter()`` will perform its depthshading
    independently.
axlim_clip : bool, default: False
    Whether to hide the scatter points outside the axes view limits.

    .. versionadded:: 3.10
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER
**kwargs
    All other keyword arguments are passed on to `~.axes.Axes.scatter`.

Returns
-------
paths : `~matplotlib.collections.PathCollection`

### Function: bar(self, left, height, zs, zdir)

**Description:** Add 2D bar(s).

Parameters
----------
left : 1D array-like
    The x coordinates of the left sides of the bars.
height : 1D array-like
    The height of the bars.
zs : float or 1D array-like, default: 0
    Z coordinate of bars; if a single value is specified, it will be
    used for all bars.
zdir : {'x', 'y', 'z'}, default: 'z'
    When plotting 2D data, the direction to use as z ('x', 'y' or 'z').
axlim_clip : bool, default: False
    Whether to hide bars with points outside the axes view limits.

    .. versionadded:: 3.10
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER
**kwargs
    Other keyword arguments are forwarded to
    `matplotlib.axes.Axes.bar`.

Returns
-------
mpl_toolkits.mplot3d.art3d.Patch3DCollection

### Function: bar3d(self, x, y, z, dx, dy, dz, color, zsort, shade, lightsource)

**Description:** Generate a 3D barplot.

This method creates three-dimensional barplot where the width,
depth, height, and color of the bars can all be uniquely set.

Parameters
----------
x, y, z : array-like
    The coordinates of the anchor point of the bars.

dx, dy, dz : float or array-like
    The width, depth, and height of the bars, respectively.

color : sequence of colors, optional
    The color of the bars can be specified globally or
    individually. This parameter can be:

    - A single color, to color all bars the same color.
    - An array of colors of length N bars, to color each bar
      independently.
    - An array of colors of length 6, to color the faces of the
      bars similarly.
    - An array of colors of length 6 * N bars, to color each face
      independently.

    When coloring the faces of the boxes specifically, this is
    the order of the coloring:

    1. -Z (bottom of box)
    2. +Z (top of box)
    3. -Y
    4. +Y
    5. -X
    6. +X

zsort : {'average', 'min', 'max'}, default: 'average'
    The z-axis sorting scheme passed onto `~.art3d.Poly3DCollection`

shade : bool, default: True
    When true, this shades the dark sides of the bars (relative
    to the plot's source of light).

lightsource : `~matplotlib.colors.LightSource`, optional
    The lightsource to use when *shade* is True.

axlim_clip : bool, default: False
    Whether to hide the bars with points outside the axes view limits.

    .. versionadded:: 3.10

data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

**kwargs
    Any additional keyword arguments are passed onto
    `~.art3d.Poly3DCollection`.

Returns
-------
collection : `~.art3d.Poly3DCollection`
    A collection of three-dimensional polygons representing the bars.

### Function: set_title(self, label, fontdict, loc)

### Function: quiver(self, X, Y, Z, U, V, W)

**Description:** Plot a 3D field of arrows.

The arguments can be array-like or scalars, so long as they can be
broadcast together. The arguments can also be masked arrays. If an
element in any of argument is masked, then that corresponding quiver
element will not be plotted.

Parameters
----------
X, Y, Z : array-like
    The x, y and z coordinates of the arrow locations (default is
    tail of arrow; see *pivot* kwarg).

U, V, W : array-like
    The x, y and z components of the arrow vectors.

length : float, default: 1
    The length of each quiver.

arrow_length_ratio : float, default: 0.3
    The ratio of the arrow head with respect to the quiver.

pivot : {'tail', 'middle', 'tip'}, default: 'tail'
    The part of the arrow that is at the grid point; the arrow
    rotates about this point, hence the name *pivot*.

normalize : bool, default: False
    Whether all arrows are normalized to have the same length, or keep
    the lengths defined by *u*, *v*, and *w*.

axlim_clip : bool, default: False
    Whether to hide arrows with points outside the axes view limits.

    .. versionadded:: 3.10

data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

**kwargs
    Any additional keyword arguments are delegated to
    :class:`.Line3DCollection`

### Function: voxels(self)

**Description:** ax.voxels([x, y, z,] /, filled, facecolors=None, edgecolors=None, **kwargs)

Plot a set of filled voxels

All voxels are plotted as 1x1x1 cubes on the axis, with
``filled[0, 0, 0]`` placed with its lower corner at the origin.
Occluded faces are not plotted.

Parameters
----------
filled : 3D np.array of bool
    A 3D array of values, with truthy values indicating which voxels
    to fill

x, y, z : 3D np.array, optional
    The coordinates of the corners of the voxels. This should broadcast
    to a shape one larger in every dimension than the shape of
    *filled*.  These can be used to plot non-cubic voxels.

    If not specified, defaults to increasing integers along each axis,
    like those returned by :func:`~numpy.indices`.
    As indicated by the ``/`` in the function signature, these
    arguments can only be passed positionally.

facecolors, edgecolors : array-like, optional
    The color to draw the faces and edges of the voxels. Can only be
    passed as keyword arguments.
    These parameters can be:

    - A single color value, to color all voxels the same color. This
      can be either a string, or a 1D RGB/RGBA array
    - ``None``, the default, to use a single color for the faces, and
      the style default for the edges.
    - A 3D `~numpy.ndarray` of color names, with each item the color
      for the corresponding voxel. The size must match the voxels.
    - A 4D `~numpy.ndarray` of RGB/RGBA data, with the components
      along the last axis.

shade : bool, default: True
    Whether to shade the facecolors.

lightsource : `~matplotlib.colors.LightSource`, optional
    The lightsource to use when *shade* is True.

axlim_clip : bool, default: False
    Whether to hide voxels with points outside the axes view limits.

    .. versionadded:: 3.10

**kwargs
    Additional keyword arguments to pass onto
    `~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.

Returns
-------
faces : dict
    A dictionary indexed by coordinate, where ``faces[i, j, k]`` is a
    `.Poly3DCollection` of the faces drawn for the voxel
    ``filled[i, j, k]``. If no faces were drawn for a given voxel,
    either because it was not asked to be drawn, or it is fully
    occluded, then ``(i, j, k) not in faces``.

Examples
--------
.. plot:: gallery/mplot3d/voxels.py
.. plot:: gallery/mplot3d/voxels_rgb.py
.. plot:: gallery/mplot3d/voxels_torus.py
.. plot:: gallery/mplot3d/voxels_numpy_logo.py

### Function: errorbar(self, x, y, z, zerr, yerr, xerr, fmt, barsabove, errorevery, ecolor, elinewidth, capsize, capthick, xlolims, xuplims, ylolims, yuplims, zlolims, zuplims, axlim_clip)

**Description:** Plot lines and/or markers with errorbars around them.

*x*/*y*/*z* define the data locations, and *xerr*/*yerr*/*zerr* define
the errorbar sizes. By default, this draws the data markers/lines as
well the errorbars. Use fmt='none' to draw errorbars only.

Parameters
----------
x, y, z : float or array-like
    The data positions.

xerr, yerr, zerr : float or array-like, shape (N,) or (2, N), optional
    The errorbar sizes:

    - scalar: Symmetric +/- values for all data points.
    - shape(N,): Symmetric +/-values for each data point.
    - shape(2, N): Separate - and + values for each bar. First row
      contains the lower errors, the second row contains the upper
      errors.
    - *None*: No errorbar.

    Note that all error arrays should have *positive* values.

fmt : str, default: ''
    The format for the data points / data lines. See `.plot` for
    details.

    Use 'none' (case-insensitive) to plot errorbars without any data
    markers.

ecolor : :mpltype:`color`, default: None
    The color of the errorbar lines.  If None, use the color of the
    line connecting the markers.

elinewidth : float, default: None
    The linewidth of the errorbar lines. If None, the linewidth of
    the current style is used.

capsize : float, default: :rc:`errorbar.capsize`
    The length of the error bar caps in points.

capthick : float, default: None
    An alias to the keyword argument *markeredgewidth* (a.k.a. *mew*).
    This setting is a more sensible name for the property that
    controls the thickness of the error bar cap in points. For
    backwards compatibility, if *mew* or *markeredgewidth* are given,
    then they will over-ride *capthick*. This may change in future
    releases.

barsabove : bool, default: False
    If True, will plot the errorbars above the plot
    symbols. Default is below.

xlolims, ylolims, zlolims : bool, default: False
    These arguments can be used to indicate that a value gives only
    lower limits. In that case a caret symbol is used to indicate
    this. *lims*-arguments may be scalars, or array-likes of the same
    length as the errors. To use limits with inverted axes,
    `~.set_xlim`, `~.set_ylim`, or `~.set_zlim` must be
    called before `errorbar`. Note the tricky parameter names: setting
    e.g. *ylolims* to True means that the y-value is a *lower* limit of
    the True value, so, only an *upward*-pointing arrow will be drawn!

xuplims, yuplims, zuplims : bool, default: False
    Same as above, but for controlling the upper limits.

errorevery : int or (int, int), default: 1
    draws error bars on a subset of the data. *errorevery* =N draws
    error bars on the points (x[::N], y[::N], z[::N]).
    *errorevery* =(start, N) draws error bars on the points
    (x[start::N], y[start::N], z[start::N]). e.g. *errorevery* =(6, 3)
    adds error bars to the data at (x[6], x[9], x[12], x[15], ...).
    Used to avoid overlapping error bars when two series share x-axis
    values.

axlim_clip : bool, default: False
    Whether to hide error bars that are outside the axes limits.

    .. versionadded:: 3.10

Returns
-------
errlines : list
    List of `~mpl_toolkits.mplot3d.art3d.Line3DCollection` instances
    each containing an errorbar line.
caplines : list
    List of `~mpl_toolkits.mplot3d.art3d.Line3D` instances each
    containing a capline object.
limmarks : list
    List of `~mpl_toolkits.mplot3d.art3d.Line3D` instances each
    containing a marker with an upper or lower limit.

Other Parameters
----------------
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

**kwargs
    All other keyword arguments for styling errorbar lines are passed
    `~mpl_toolkits.mplot3d.art3d.Line3DCollection`.

Examples
--------
.. plot:: gallery/mplot3d/errorbar3d.py

### Function: get_tightbbox(self, renderer)

### Function: stem(self, x, y, z)

**Description:** Create a 3D stem plot.

A stem plot draws lines perpendicular to a baseline, and places markers
at the heads. By default, the baseline is defined by *x* and *y*, and
stems are drawn vertically from *bottom* to *z*.

Parameters
----------
x, y, z : array-like
    The positions of the heads of the stems. The stems are drawn along
    the *orientation*-direction from the baseline at *bottom* (in the
    *orientation*-coordinate) to the heads. By default, the *x* and *y*
    positions are used for the baseline and *z* for the head position,
    but this can be changed by *orientation*.

linefmt : str, default: 'C0-'
    A string defining the properties of the vertical lines. Usually,
    this will be a color or a color and a linestyle:

    =========  =============
    Character  Line Style
    =========  =============
    ``'-'``    solid line
    ``'--'``   dashed line
    ``'-.'``   dash-dot line
    ``':'``    dotted line
    =========  =============

    Note: While it is technically possible to specify valid formats
    other than color or color and linestyle (e.g. 'rx' or '-.'), this
    is beyond the intention of the method and will most likely not
    result in a reasonable plot.

markerfmt : str, default: 'C0o'
    A string defining the properties of the markers at the stem heads.

basefmt : str, default: 'C3-'
    A format string defining the properties of the baseline.

bottom : float, default: 0
    The position of the baseline, in *orientation*-coordinates.

label : str, optional
    The label to use for the stems in legends.

orientation : {'x', 'y', 'z'}, default: 'z'
    The direction along which stems are drawn.

axlim_clip : bool, default: False
    Whether to hide stems that are outside the axes limits.

    .. versionadded:: 3.10

data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

Returns
-------
`.StemContainer`
    The container may be treated like a tuple
    (*markerline*, *stemlines*, *baseline*)

Examples
--------
.. plot:: gallery/mplot3d/stem3d_demo.py

### Function: __init__(self, scalar, vector)

### Function: __neg__(self)

### Function: __mul__(self, other)

**Description:** Product of two quaternions
i*i = j*j = k*k = i*j*k = -1
Quaternion multiplication can be expressed concisely
using scalar and vector parts,
see <https://en.wikipedia.org/wiki/Quaternion#Scalar_and_vector_parts>

### Function: conjugate(self)

**Description:** The conjugate quaternion -(1/2)*(q+i*q*i+j*q*j+k*q*k)

### Function: norm(self)

**Description:** The 2-norm, q*q', a scalar

### Function: normalize(self)

**Description:** Scaling such that norm equals 1

### Function: reciprocal(self)

**Description:** The reciprocal, 1/q = q'/(q*q') = q' / norm(q)

### Function: __div__(self, other)

### Function: rotate(self, v)

### Function: __eq__(self, other)

### Function: __repr__(self)

### Function: rotate_from_to(cls, r1, r2)

**Description:** The quaternion for the shortest rotation from vector r1 to vector r2
i.e., q = sqrt(r2*r1'), normalized.
If r1 and r2 are antiparallel, then the result is ambiguous;
a normal vector will be returned, and a warning will be issued.

### Function: from_cardan_angles(cls, elev, azim, roll)

**Description:** Converts the angles to a quaternion
    q = exp((roll/2)*e_x)*exp((elev/2)*e_y)*exp((-azim/2)*e_z)
i.e., the angles are a kind of Tait-Bryan angles, -z,y',x".
The angles should be given in radians, not degrees.

### Function: as_cardan_angles(self)

**Description:** The inverse of `from_cardan_angles()`.
Note that the angles returned are in radians, not degrees.
The angles are not sensitive to the quaternion's norm().

### Function: calc_arrows(UVW)

### Function: _broadcast_color_arg(color, name)

### Function: permutation_matrices(n)

**Description:** Generate cyclic permutation matrices.

### Function: _apply_mask(arrays, mask)

### Function: _extract_errs(err, data, lomask, himask)

### Function: _digout_minmax(err_arr, coord_label)

### Function: voxels(__x, __y, __z, filled)

### Function: voxels(filled)
