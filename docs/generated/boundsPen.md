## AI Summary

A file named boundsPen.py.


## Class: ControlBoundsPen

**Description:** Pen to calculate the "control bounds" of a shape. This is the
bounding box of all control points, so may be larger than the
actual bounding box if there are curves that don't have points
on their extremes.

When the shape has been drawn, the bounds are available as the
``bounds`` attribute of the pen object. It's a 4-tuple::

        (xMin, yMin, xMax, yMax).

If ``ignoreSinglePoints`` is True, single points are ignored.

## Class: BoundsPen

**Description:** Pen to calculate the bounds of a shape. It calculates the
correct bounds even when the shape contains curves that don't
have points on their extremes. This is somewhat slower to compute
than the "control bounds".

When the shape has been drawn, the bounds are available as the
``bounds`` attribute of the pen object. It's a 4-tuple::

        (xMin, yMin, xMax, yMax)

### Function: __init__(self, glyphSet, ignoreSinglePoints)

### Function: init(self)

### Function: _moveTo(self, pt)

### Function: _addMoveTo(self)

### Function: _lineTo(self, pt)

### Function: _curveToOne(self, bcp1, bcp2, pt)

### Function: _qCurveToOne(self, bcp, pt)

### Function: _curveToOne(self, bcp1, bcp2, pt)

### Function: _qCurveToOne(self, bcp, pt)
