## AI Summary

A file named reverseContourPen.py.


## Class: ReverseContourPen

**Description:** Filter pen that passes outline data to another pen, but reversing
the winding direction of all contours. Components are simply passed
through unchanged.

Closed contours are reversed in such a way that the first point remains
the first point.

### Function: reversedContour(contour, outputImpliedClosingLine)

**Description:** Generator that takes a list of pen's (operator, operands) tuples,
and yields them with the winding direction reversed.

### Function: __init__(self, outPen, outputImpliedClosingLine)

### Function: filterContour(self, contour)
