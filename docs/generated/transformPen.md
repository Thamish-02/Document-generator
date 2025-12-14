## AI Summary

A file named transformPen.py.


## Class: TransformPen

**Description:** Pen that transforms all coordinates using a Affine transformation,
and passes them to another pen.

## Class: TransformPointPen

**Description:** PointPen that transforms all coordinates using a Affine transformation,
and passes them to another PointPen.

For example::

    >>> from fontTools.pens.recordingPen import RecordingPointPen
    >>> rec = RecordingPointPen()
    >>> pen = TransformPointPen(rec, (2, 0, 0, 2, -10, 5))
    >>> v = iter(rec.value)
    >>> pen.beginPath(identifier="contour-0")
    >>> next(v)
    ('beginPath', (), {'identifier': 'contour-0'})

    >>> pen.addPoint((100, 100), "line")
    >>> next(v)
    ('addPoint', ((190, 205), 'line', False, None), {})

    >>> pen.endPath()
    >>> next(v)
    ('endPath', (), {})

    >>> pen.addComponent("a", (1, 0, 0, 1, -10, 5), identifier="component-0")
    >>> next(v)
    ('addComponent', ('a', <Transform [2 0 0 2 -30 15]>), {'identifier': 'component-0'})

### Function: __init__(self, outPen, transformation)

**Description:** The 'outPen' argument is another pen object. It will receive the
transformed coordinates. The 'transformation' argument can either
be a six-tuple, or a fontTools.misc.transform.Transform object.

### Function: moveTo(self, pt)

### Function: lineTo(self, pt)

### Function: curveTo(self)

### Function: qCurveTo(self)

### Function: _transformPoints(self, points)

### Function: closePath(self)

### Function: endPath(self)

### Function: addComponent(self, glyphName, transformation)

### Function: __init__(self, outPointPen, transformation)

**Description:** The 'outPointPen' argument is another point pen object.
It will receive the transformed coordinates.
The 'transformation' argument can either be a six-tuple, or a
fontTools.misc.transform.Transform object.

### Function: addPoint(self, pt, segmentType, smooth, name)

### Function: addComponent(self, baseGlyphName, transformation)
