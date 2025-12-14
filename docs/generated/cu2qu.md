## AI Summary

A file named cu2qu.py.


### Function: dot(v1, v2)

**Description:** Return the dot product of two vectors.

Args:
    v1 (complex): First vector.
    v2 (complex): Second vector.

Returns:
    double: Dot product.

### Function: _complex_div_by_real(z, den)

**Description:** Divide complex by real using Python's method (two separate divisions).

This ensures bit-exact compatibility with Python's complex division,
avoiding C's multiply-by-reciprocal optimization that can cause 1 ULP differences
on some platforms/compilers (e.g. clang on macOS arm64).

https://github.com/fonttools/fonttools/issues/3928

### Function: calc_cubic_points(a, b, c, d)

### Function: calc_cubic_parameters(p0, p1, p2, p3)

### Function: split_cubic_into_n_iter(p0, p1, p2, p3, n)

**Description:** Split a cubic Bezier into n equal parts.

Splits the curve into `n` equal parts by curve time.
(t=0..1/n, t=1/n..2/n, ...)

Args:
    p0 (complex): Start point of curve.
    p1 (complex): First handle of curve.
    p2 (complex): Second handle of curve.
    p3 (complex): End point of curve.

Returns:
    An iterator yielding the control points (four complex values) of the
    subcurves.

### Function: _split_cubic_into_n_gen(p0, p1, p2, p3, n)

### Function: split_cubic_into_two(p0, p1, p2, p3)

**Description:** Split a cubic Bezier into two equal parts.

Splits the curve into two equal parts at t = 0.5

Args:
    p0 (complex): Start point of curve.
    p1 (complex): First handle of curve.
    p2 (complex): Second handle of curve.
    p3 (complex): End point of curve.

Returns:
    tuple: Two cubic Beziers (each expressed as a tuple of four complex
    values).

### Function: split_cubic_into_three(p0, p1, p2, p3)

**Description:** Split a cubic Bezier into three equal parts.

Splits the curve into three equal parts at t = 1/3 and t = 2/3

Args:
    p0 (complex): Start point of curve.
    p1 (complex): First handle of curve.
    p2 (complex): Second handle of curve.
    p3 (complex): End point of curve.

Returns:
    tuple: Three cubic Beziers (each expressed as a tuple of four complex
    values).

### Function: cubic_approx_control(t, p0, p1, p2, p3)

**Description:** Approximate a cubic Bezier using a quadratic one.

Args:
    t (double): Position of control point.
    p0 (complex): Start point of curve.
    p1 (complex): First handle of curve.
    p2 (complex): Second handle of curve.
    p3 (complex): End point of curve.

Returns:
    complex: Location of candidate control point on quadratic curve.

### Function: calc_intersect(a, b, c, d)

**Description:** Calculate the intersection of two lines.

Args:
    a (complex): Start point of first line.
    b (complex): End point of first line.
    c (complex): Start point of second line.
    d (complex): End point of second line.

Returns:
    complex: Location of intersection if one present, ``complex(NaN,NaN)``
    if no intersection was found.

### Function: cubic_farthest_fit_inside(p0, p1, p2, p3, tolerance)

**Description:** Check if a cubic Bezier lies within a given distance of the origin.

"Origin" means *the* origin (0,0), not the start of the curve. Note that no
checks are made on the start and end positions of the curve; this function
only checks the inside of the curve.

Args:
    p0 (complex): Start point of curve.
    p1 (complex): First handle of curve.
    p2 (complex): Second handle of curve.
    p3 (complex): End point of curve.
    tolerance (double): Distance from origin.

Returns:
    bool: True if the cubic Bezier ``p`` entirely lies within a distance
    ``tolerance`` of the origin, False otherwise.

### Function: cubic_approx_quadratic(cubic, tolerance)

**Description:** Approximate a cubic Bezier with a single quadratic within a given tolerance.

Args:
    cubic (sequence): Four complex numbers representing control points of
        the cubic Bezier curve.
    tolerance (double): Permitted deviation from the original curve.

Returns:
    Three complex numbers representing control points of the quadratic
    curve if it fits within the given tolerance, or ``None`` if no suitable
    curve could be calculated.

### Function: cubic_approx_spline(cubic, n, tolerance, all_quadratic)

**Description:** Approximate a cubic Bezier curve with a spline of n quadratics.

Args:
    cubic (sequence): Four complex numbers representing control points of
        the cubic Bezier curve.
    n (int): Number of quadratic Bezier curves in the spline.
    tolerance (double): Permitted deviation from the original curve.

Returns:
    A list of ``n+2`` complex numbers, representing control points of the
    quadratic spline if it fits within the given tolerance, or ``None`` if
    no suitable spline could be calculated.

### Function: curve_to_quadratic(curve, max_err, all_quadratic)

**Description:** Approximate a cubic Bezier curve with a spline of n quadratics.

Args:
    cubic (sequence): Four 2D tuples representing control points of
        the cubic Bezier curve.
    max_err (double): Permitted deviation from the original curve.
    all_quadratic (bool): If True (default) returned value is a
        quadratic spline. If False, it's either a single quadratic
        curve or a single cubic curve.

Returns:
    If all_quadratic is True: A list of 2D tuples, representing
    control points of the quadratic spline if it fits within the
    given tolerance, or ``None`` if no suitable spline could be
    calculated.

    If all_quadratic is False: Either a quadratic curve (if length
    of output is 3), or a cubic curve (if length of output is 4).

### Function: curves_to_quadratic(curves, max_errors, all_quadratic)

**Description:** Return quadratic Bezier splines approximating the input cubic Beziers.

Args:
    curves: A sequence of *n* curves, each curve being a sequence of four
        2D tuples.
    max_errors: A sequence of *n* floats representing the maximum permissible
        deviation from each of the cubic Bezier curves.
    all_quadratic (bool): If True (default) returned values are a
        quadratic spline. If False, they are either a single quadratic
        curve or a single cubic curve.

Example::

    >>> curves_to_quadratic( [
    ...   [ (50,50), (100,100), (150,100), (200,50) ],
    ...   [ (75,50), (120,100), (150,75),  (200,60) ]
    ... ], [1,1] )
    [[(50.0, 50.0), (75.0, 75.0), (125.0, 91.66666666666666), (175.0, 75.0), (200.0, 50.0)], [(75.0, 50.0), (97.5, 75.0), (135.41666666666666, 82.08333333333333), (175.0, 67.5), (200.0, 60.0)]]

The returned splines have "implied oncurve points" suitable for use in
TrueType ``glif`` outlines - i.e. in the first spline returned above,
the first quadratic segment runs from (50,50) to
( (75 + 125)/2 , (120 + 91.666..)/2 ) = (100, 83.333...).

Returns:
    If all_quadratic is True, a list of splines, each spline being a list
    of 2D tuples.

    If all_quadratic is False, a list of curves, each curve being a quadratic
    (length 3), or cubic (length 4).

Raises:
    fontTools.cu2qu.Errors.ApproxNotFoundError: if no suitable approximation
    can be found for all curves with the given parameters.
