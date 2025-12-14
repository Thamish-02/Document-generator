## AI Summary

A file named cu2quPen.py.


## Class: Cu2QuPen

**Description:** A filter pen to convert cubic bezier curves to quadratic b-splines
using the FontTools SegmentPen protocol.

Args:

    other_pen: another SegmentPen used to draw the transformed outline.
    max_err: maximum approximation error in font units. For optimal results,
        if you know the UPEM of the font, we recommend setting this to a
        value equal, or close to UPEM / 1000.
    reverse_direction: flip the contours' direction but keep starting point.
    stats: a dictionary counting the point numbers of quadratic segments.
    all_quadratic: if True (default), only quadratic b-splines are generated.
        if False, quadratic curves or cubic curves are generated depending
        on which one is more economical.

## Class: Cu2QuPointPen

**Description:** A filter pen to convert cubic bezier curves to quadratic b-splines
using the FontTools PointPen protocol.

Args:
    other_point_pen: another PointPen used to draw the transformed outline.
    max_err: maximum approximation error in font units. For optimal results,
        if you know the UPEM of the font, we recommend setting this to a
        value equal, or close to UPEM / 1000.
    reverse_direction: reverse the winding direction of all contours.
    stats: a dictionary counting the point numbers of quadratic segments.
    all_quadratic: if True (default), only quadratic b-splines are generated.
        if False, quadratic curves or cubic curves are generated depending
        on which one is more economical.

## Class: Cu2QuMultiPen

**Description:** A filter multi-pen to convert cubic bezier curves to quadratic b-splines
in a interpolation-compatible manner, using the FontTools SegmentPen protocol.

Args:

    other_pens: list of SegmentPens used to draw the transformed outlines.
    max_err: maximum approximation error in font units. For optimal results,
        if you know the UPEM of the font, we recommend setting this to a
        value equal, or close to UPEM / 1000.
    reverse_direction: flip the contours' direction but keep starting point.

This pen does not follow the normal SegmentPen protocol. Instead, its
moveTo/lineTo/qCurveTo/curveTo methods take a list of tuples that are
arguments that would normally be passed to a SegmentPen, one item for
each of the pens in other_pens.

### Function: __init__(self, other_pen, max_err, reverse_direction, stats, all_quadratic)

### Function: _convert_curve(self, pt1, pt2, pt3)

### Function: curveTo(self)

### Function: __init__(self, other_point_pen, max_err, reverse_direction, stats, all_quadratic)

### Function: _flushContour(self, segments)

### Function: _split_super_bezier_segments(self, points)

### Function: _drawPoints(self, segments)

### Function: addComponent(self, baseGlyphName, transformation)

### Function: __init__(self, other_pens, max_err, reverse_direction)

### Function: _check_contour_is_open(self)

### Function: _check_contour_is_closed(self)

### Function: _add_moveTo(self)

### Function: moveTo(self, pts)

### Function: lineTo(self, pts)

### Function: qCurveTo(self, pointsList)

### Function: _curves_to_quadratic(self, pointsList)

### Function: curveTo(self, pointsList)

### Function: closePath(self)

### Function: endPath(self)

### Function: addComponent(self, glyphName, transformations)
