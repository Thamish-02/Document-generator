## AI Summary

A file named t2CharStringPen.py.


## Class: T2CharStringPen

**Description:** Pen to draw Type 2 CharStrings.

The 'roundTolerance' argument controls the rounding of point coordinates.
It is defined as the maximum absolute difference between the original
float and the rounded integer value.
The default tolerance of 0.5 means that all floats are rounded to integer;
a value of 0 disables rounding; values in between will only round floats
which are close to their integral part within the tolerated range.

### Function: __init__(self, width, glyphSet, roundTolerance, CFF2)

### Function: _p(self, pt)

### Function: _moveTo(self, pt)

### Function: _lineTo(self, pt)

### Function: _curveToOne(self, pt1, pt2, pt3)

### Function: _closePath(self)

### Function: _endPath(self)

### Function: getCharString(self, private, globalSubrs, optimize)
