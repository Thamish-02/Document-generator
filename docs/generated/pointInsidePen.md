## AI Summary

A file named pointInsidePen.py.


## Class: PointInsidePen

**Description:** This pen implements "point inside" testing: to test whether
a given point lies inside the shape (black) or outside (white).
Instances of this class can be recycled, as long as the
setTestPoint() method is used to set the new point to test.

:Example:
    .. code-block::

        pen = PointInsidePen(glyphSet, (100, 200))
        outline.draw(pen)
        isInside = pen.getResult()

Both the even-odd algorithm and the non-zero-winding-rule
algorithm are implemented. The latter is the default, specify
True for the evenOdd argument of __init__ or setTestPoint
to use the even-odd algorithm.

### Function: __init__(self, glyphSet, testPoint, evenOdd)

### Function: setTestPoint(self, testPoint, evenOdd)

**Description:** Set the point to test. Call this _before_ the outline gets drawn.

### Function: getWinding(self)

### Function: getResult(self)

**Description:** After the shape has been drawn, getResult() returns True if the test
point lies within the (black) shape, and False if it doesn't.

### Function: _addIntersection(self, goingUp)

### Function: _moveTo(self, point)

### Function: _lineTo(self, point)

### Function: _curveToOne(self, bcp1, bcp2, point)

### Function: _qCurveToOne_unfinished(self, bcp, point)

### Function: _closePath(self)

### Function: _endPath(self)

**Description:** Insideness is not defined for open contours.
