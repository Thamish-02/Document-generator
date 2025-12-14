## AI Summary

A file named qu2cu.py.


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

### Function: elevate_quadratic(p0, p1, p2)

**Description:** Given a quadratic bezier curve, return its degree-elevated cubic.

### Function: merge_curves(curves, start, n)

**Description:** Give a cubic-Bezier spline, reconstruct one cubic-Bezier
that has the same endpoints and tangents and approxmates
the spline.

### Function: add_implicit_on_curves(p)

### Function: quadratic_to_curves(quads, max_err, all_cubic)

**Description:** Converts a connecting list of quadratic splines to a list of quadratic
and cubic curves.

A quadratic spline is specified as a list of points.  Either each point is
a 2-tuple of X,Y coordinates, or each point is a complex number with
real/imaginary components representing X,Y coordinates.

The first and last points are on-curve points and the rest are off-curve
points, with an implied on-curve point in the middle between every two
consequtive off-curve points.

Returns:
    The output is a list of tuples of points. Points are represented
    in the same format as the input, either as 2-tuples or complex numbers.

    Each tuple is either of length three, for a quadratic curve, or four,
    for a cubic curve.  Each curve's last point is the same as the next
    curve's first point.

Args:
    quads: quadratic splines

    max_err: absolute error tolerance; defaults to 0.5

    all_cubic: if True, only cubic curves are generated; defaults to False

### Function: spline_to_curves(q, costs, tolerance, all_cubic)

**Description:** q: quadratic spline with alternating on-curve / off-curve points.

costs: cumulative list of encoding cost of q in terms of number of
  points that need to be encoded.  Implied on-curve points do not
  contribute to the cost. If all points need to be encoded, then
  costs will be range(1, len(q)+1).

### Function: main()
