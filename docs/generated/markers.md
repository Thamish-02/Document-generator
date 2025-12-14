## AI Summary

A file named markers.py.


## Class: MarkerStyle

**Description:** A class representing marker types.

Instances are immutable. If you need to change anything, create a new
instance.

Attributes
----------
markers : dict
    All known markers.
filled_markers : tuple
    All known filled markers. This is a subset of *markers*.
fillstyles : tuple
    The supported fillstyles.

### Function: __init__(self, marker, fillstyle, transform, capstyle, joinstyle)

**Description:** Parameters
----------
marker : str, array-like, Path, MarkerStyle
    - Another instance of `MarkerStyle` copies the details of that *marker*.
    - For other possible marker values, see the module docstring
      `matplotlib.markers`.

fillstyle : str, default: :rc:`markers.fillstyle`
    One of 'full', 'left', 'right', 'bottom', 'top', 'none'.

transform : `~matplotlib.transforms.Transform`, optional
    Transform that will be combined with the native transform of the
    marker.

capstyle : `.CapStyle` or %(CapStyle)s, optional
    Cap style that will override the default cap style of the marker.

joinstyle : `.JoinStyle` or %(JoinStyle)s, optional
    Join style that will override the default join style of the marker.

### Function: _recache(self)

### Function: __bool__(self)

### Function: is_filled(self)

### Function: get_fillstyle(self)

### Function: _set_fillstyle(self, fillstyle)

**Description:** Set the fillstyle.

Parameters
----------
fillstyle : {'full', 'left', 'right', 'bottom', 'top', 'none'}
    The part of the marker surface that is colored with
    markerfacecolor.

### Function: get_joinstyle(self)

### Function: get_capstyle(self)

### Function: get_marker(self)

### Function: _set_marker(self, marker)

**Description:** Set the marker.

Parameters
----------
marker : str, array-like, Path, MarkerStyle
    - Another instance of `MarkerStyle` copies the details of that *marker*.
    - For other possible marker values see the module docstring
      `matplotlib.markers`.

### Function: get_path(self)

**Description:** Return a `.Path` for the primary part of the marker.

For unfilled markers this is the whole marker, for filled markers,
this is the area to be drawn with *markerfacecolor*.

### Function: get_transform(self)

**Description:** Return the transform to be applied to the `.Path` from
`MarkerStyle.get_path()`.

### Function: get_alt_path(self)

**Description:** Return a `.Path` for the alternate part of the marker.

For unfilled markers, this is *None*; for filled markers, this is the
area to be drawn with *markerfacecoloralt*.

### Function: get_alt_transform(self)

**Description:** Return the transform to be applied to the `.Path` from
`MarkerStyle.get_alt_path()`.

### Function: get_snap_threshold(self)

### Function: get_user_transform(self)

**Description:** Return user supplied part of marker transform.

### Function: transformed(self, transform)

**Description:** Return a new version of this marker with the transform applied.

Parameters
----------
transform : `~matplotlib.transforms.Affine2D`
    Transform will be combined with current user supplied transform.

### Function: rotated(self)

**Description:** Return a new version of this marker rotated by specified angle.

Parameters
----------
deg : float, optional
    Rotation angle in degrees.

rad : float, optional
    Rotation angle in radians.

.. note:: You must specify exactly one of deg or rad.

### Function: scaled(self, sx, sy)

**Description:** Return new marker scaled by specified scale factors.

If *sy* is not given, the same scale is applied in both the *x*- and
*y*-directions.

Parameters
----------
sx : float
    *X*-direction scaling factor.
sy : float, optional
    *Y*-direction scaling factor.

### Function: _set_nothing(self)

### Function: _set_custom_marker(self, path)

### Function: _set_path_marker(self)

### Function: _set_vertices(self)

### Function: _set_tuple_marker(self)

### Function: _set_mathtext_path(self)

**Description:** Draw mathtext markers '$...$' using `.TextPath` object.

Submitted by tcb

### Function: _half_fill(self)

### Function: _set_circle(self, size)

### Function: _set_point(self)

### Function: _set_pixel(self)

### Function: _set_triangle(self, rot, skip)

### Function: _set_triangle_up(self)

### Function: _set_triangle_down(self)

### Function: _set_triangle_left(self)

### Function: _set_triangle_right(self)

### Function: _set_square(self)

### Function: _set_diamond(self)

### Function: _set_thin_diamond(self)

### Function: _set_pentagon(self)

### Function: _set_star(self)

### Function: _set_hexagon1(self)

### Function: _set_hexagon2(self)

### Function: _set_octagon(self)

### Function: _set_vline(self)

### Function: _set_hline(self)

### Function: _set_tickleft(self)

### Function: _set_tickright(self)

### Function: _set_tickup(self)

### Function: _set_tickdown(self)

### Function: _set_tri_down(self)

### Function: _set_tri_up(self)

### Function: _set_tri_left(self)

### Function: _set_tri_right(self)

### Function: _set_caretdown(self)

### Function: _set_caretup(self)

### Function: _set_caretleft(self)

### Function: _set_caretright(self)

### Function: _set_caretdownbase(self)

### Function: _set_caretupbase(self)

### Function: _set_caretleftbase(self)

### Function: _set_caretrightbase(self)

### Function: _set_plus(self)

### Function: _set_x(self)

### Function: _set_plus_filled(self)

### Function: _set_x_filled(self)
