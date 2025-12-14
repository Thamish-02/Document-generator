## AI Summary

A file named bezier.py.


### Function: _comb(n, k)

## Class: NonIntersectingPathException

### Function: get_intersection(cx1, cy1, cos_t1, sin_t1, cx2, cy2, cos_t2, sin_t2)

**Description:** Return the intersection between the line through (*cx1*, *cy1*) at angle
*t1* and the line through (*cx2*, *cy2*) at angle *t2*.

### Function: get_normal_points(cx, cy, cos_t, sin_t, length)

**Description:** For a line passing through (*cx*, *cy*) and having an angle *t*, return
locations of the two points located along its perpendicular line at the
distance of *length*.

### Function: _de_casteljau1(beta, t)

### Function: split_de_casteljau(beta, t)

**Description:** Split a Bézier segment defined by its control points *beta* into two
separate segments divided at *t* and return their control points.

### Function: find_bezier_t_intersecting_with_closedpath(bezier_point_at_t, inside_closedpath, t0, t1, tolerance)

**Description:** Find the intersection of the Bézier curve with a closed path.

The intersection point *t* is approximated by two parameters *t0*, *t1*
such that *t0* <= *t* <= *t1*.

Search starts from *t0* and *t1* and uses a simple bisecting algorithm
therefore one of the end points must be inside the path while the other
doesn't. The search stops when the distance of the points parametrized by
*t0* and *t1* gets smaller than the given *tolerance*.

Parameters
----------
bezier_point_at_t : callable
    A function returning x, y coordinates of the Bézier at parameter *t*.
    It must have the signature::

        bezier_point_at_t(t: float) -> tuple[float, float]

inside_closedpath : callable
    A function returning True if a given point (x, y) is inside the
    closed path. It must have the signature::

        inside_closedpath(point: tuple[float, float]) -> bool

t0, t1 : float
    Start parameters for the search.

tolerance : float
    Maximal allowed distance between the final points.

Returns
-------
t0, t1 : float
    The Bézier path parameters.

## Class: BezierSegment

**Description:** A d-dimensional Bézier segment.

Parameters
----------
control_points : (N, d) array
    Location of the *N* control points.

### Function: split_bezier_intersecting_with_closedpath(bezier, inside_closedpath, tolerance)

**Description:** Split a Bézier curve into two at the intersection with a closed path.

Parameters
----------
bezier : (N, 2) array-like
    Control points of the Bézier segment. See `.BezierSegment`.
inside_closedpath : callable
    A function returning True if a given point (x, y) is inside the
    closed path. See also `.find_bezier_t_intersecting_with_closedpath`.
tolerance : float
    The tolerance for the intersection. See also
    `.find_bezier_t_intersecting_with_closedpath`.

Returns
-------
left, right
    Lists of control points for the two Bézier segments.

### Function: split_path_inout(path, inside, tolerance, reorder_inout)

**Description:** Divide a path into two segments at the point where ``inside(x, y)`` becomes
False.

### Function: inside_circle(cx, cy, r)

**Description:** Return a function that checks whether a point is in a circle with center
(*cx*, *cy*) and radius *r*.

The returned function has the signature::

    f(xy: tuple[float, float]) -> bool

### Function: get_cos_sin(x0, y0, x1, y1)

### Function: check_if_parallel(dx1, dy1, dx2, dy2, tolerance)

**Description:** Check if two lines are parallel.

Parameters
----------
dx1, dy1, dx2, dy2 : float
    The gradients *dy*/*dx* of the two lines.
tolerance : float
    The angular tolerance in radians up to which the lines are considered
    parallel.

Returns
-------
is_parallel
    - 1 if two lines are parallel in same direction.
    - -1 if two lines are parallel in opposite direction.
    - False otherwise.

### Function: get_parallels(bezier2, width)

**Description:** Given the quadratic Bézier control points *bezier2*, returns
control points of quadratic Bézier lines roughly parallel to given
one separated by *width*.

### Function: find_control_points(c1x, c1y, mmx, mmy, c2x, c2y)

**Description:** Find control points of the Bézier curve passing through (*c1x*, *c1y*),
(*mmx*, *mmy*), and (*c2x*, *c2y*), at parametric values 0, 0.5, and 1.

### Function: make_wedged_bezier2(bezier2, width, w1, wm, w2)

**Description:** Being similar to `get_parallels`, returns control points of two quadratic
Bézier lines having a width roughly parallel to given one separated by
*width*.

### Function: __init__(self, control_points)

### Function: __call__(self, t)

**Description:** Evaluate the Bézier curve at point(s) *t* in [0, 1].

Parameters
----------
t : (k,) array-like
    Points at which to evaluate the curve.

Returns
-------
(k, d) array
    Value of the curve for each point in *t*.

### Function: point_at_t(self, t)

**Description:** Evaluate the curve at a single point, returning a tuple of *d* floats.

### Function: control_points(self)

**Description:** The control points of the curve.

### Function: dimension(self)

**Description:** The dimension of the curve.

### Function: degree(self)

**Description:** Degree of the polynomial. One less the number of control points.

### Function: polynomial_coefficients(self)

**Description:** The polynomial coefficients of the Bézier curve.

.. warning:: Follows opposite convention from `numpy.polyval`.

Returns
-------
(n+1, d) array
    Coefficients after expanding in polynomial basis, where :math:`n`
    is the degree of the Bézier curve and :math:`d` its dimension.
    These are the numbers (:math:`C_j`) such that the curve can be
    written :math:`\sum_{j=0}^n C_j t^j`.

Notes
-----
The coefficients are calculated as

.. math::

    {n \choose j} \sum_{i=0}^j (-1)^{i+j} {j \choose i} P_i

where :math:`P_i` are the control points of the curve.

### Function: axis_aligned_extrema(self)

**Description:** Return the dimension and location of the curve's interior extrema.

The extrema are the points along the curve where one of its partial
derivatives is zero.

Returns
-------
dims : array of int
    Index :math:`i` of the partial derivative which is zero at each
    interior extrema.
dzeros : array of float
    Of same size as dims. The :math:`t` such that :math:`d/dx_i B(t) =
    0`

### Function: _f(xy)
