## AI Summary

A file named qu2cuPen.py.


## Class: Qu2CuPen

**Description:** A filter pen to convert quadratic bezier splines to cubic curves
using the FontTools SegmentPen protocol.

Args:

    other_pen: another SegmentPen used to draw the transformed outline.
    max_err: maximum approximation error in font units. For optimal results,
        if you know the UPEM of the font, we recommend setting this to a
        value equal, or close to UPEM / 1000.
    reverse_direction: flip the contours' direction but keep starting point.
    stats: a dictionary counting the point numbers of cubic segments.

### Function: __init__(self, other_pen, max_err, all_cubic, reverse_direction, stats)

### Function: _quadratics_to_curve(self, q)

### Function: filterContour(self, contour)
