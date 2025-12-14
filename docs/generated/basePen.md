## AI Summary

A file named basePen.py.


## Class: PenError

**Description:** Represents an error during penning.

## Class: OpenContourError

## Class: AbstractPen

## Class: NullPen

**Description:** A pen that does nothing.

## Class: LoggingPen

**Description:** A pen with a ``log`` property (see fontTools.misc.loggingTools.LogMixin)

## Class: MissingComponentError

**Description:** Indicates a component pointing to a non-existent glyph in the glyphset.

## Class: DecomposingPen

**Description:** Implements a 'addComponent' method that decomposes components
(i.e. draws them onto self as simple contours).
It can also be used as a mixin class (e.g. see ContourRecordingPen).

You must override moveTo, lineTo, curveTo and qCurveTo. You may
additionally override closePath, endPath and addComponent.

By default a warning message is logged when a base glyph is missing;
set the class variable ``skipMissingComponents`` to False if you want
all instances of a sub-class to raise a :class:`MissingComponentError`
exception by default.

## Class: BasePen

**Description:** Base class for drawing pens. You must override _moveTo, _lineTo and
_curveToOne. You may additionally override _closePath, _endPath,
addComponent, addVarComponent, and/or _qCurveToOne. You should not
override any other methods.

### Function: decomposeSuperBezierSegment(points)

**Description:** Split the SuperBezier described by 'points' into a list of regular
bezier segments. The 'points' argument must be a sequence with length
3 or greater, containing (x, y) coordinates. The last point is the
destination on-curve point, the rest of the points are off-curve points.
The start point should not be supplied.

This function returns a list of (pt1, pt2, pt3) tuples, which each
specify a regular curveto-style bezier segment.

### Function: decomposeQuadraticSegment(points)

**Description:** Split the quadratic curve segment described by 'points' into a list
of "atomic" quadratic segments. The 'points' argument must be a sequence
with length 2 or greater, containing (x, y) coordinates. The last point
is the destination on-curve point, the rest of the points are off-curve
points. The start point should not be supplied.

This function returns a list of (pt1, pt2) tuples, which each specify a
plain quadratic bezier segment.

## Class: _TestPen

**Description:** Test class that prints PostScript to stdout.

### Function: moveTo(self, pt)

**Description:** Begin a new sub path, set the current point to 'pt'. You must
end each sub path with a call to pen.closePath() or pen.endPath().

### Function: lineTo(self, pt)

**Description:** Draw a straight line from the current point to 'pt'.

### Function: curveTo(self)

**Description:** Draw a cubic bezier with an arbitrary number of control points.

The last point specified is on-curve, all others are off-curve
(control) points. If the number of control points is > 2, the
segment is split into multiple bezier segments. This works
like this:

Let n be the number of control points (which is the number of
arguments to this call minus 1). If n==2, a plain vanilla cubic
bezier is drawn. If n==1, we fall back to a quadratic segment and
if n==0 we draw a straight line. It gets interesting when n>2:
n-1 PostScript-style cubic segments will be drawn as if it were
one curve. See decomposeSuperBezierSegment().

The conversion algorithm used for n>2 is inspired by NURB
splines, and is conceptually equivalent to the TrueType "implied
points" principle. See also decomposeQuadraticSegment().

### Function: qCurveTo(self)

**Description:** Draw a whole string of quadratic curve segments.

The last point specified is on-curve, all others are off-curve
points.

This method implements TrueType-style curves, breaking up curves
using 'implied points': between each two consequtive off-curve points,
there is one implied point exactly in the middle between them. See
also decomposeQuadraticSegment().

The last argument (normally the on-curve point) may be None.
This is to support contours that have NO on-curve points (a rarely
seen feature of TrueType outlines).

### Function: closePath(self)

**Description:** Close the current sub path. You must call either pen.closePath()
or pen.endPath() after each sub path.

### Function: endPath(self)

**Description:** End the current sub path, but don't close it. You must call
either pen.closePath() or pen.endPath() after each sub path.

### Function: addComponent(self, glyphName, transformation)

**Description:** Add a sub glyph. The 'transformation' argument must be a 6-tuple
containing an affine transformation, or a Transform object from the
fontTools.misc.transform module. More precisely: it should be a
sequence containing 6 numbers.

### Function: addVarComponent(self, glyphName, transformation, location)

**Description:** Add a VarComponent sub glyph. The 'transformation' argument
must be a DecomposedTransform from the fontTools.misc.transform module,
and the 'location' argument must be a dictionary mapping axis tags
to their locations.

### Function: moveTo(self, pt)

### Function: lineTo(self, pt)

### Function: curveTo(self)

### Function: qCurveTo(self)

### Function: closePath(self)

### Function: endPath(self)

### Function: addComponent(self, glyphName, transformation)

### Function: addVarComponent(self, glyphName, transformation, location)

### Function: __init__(self, glyphSet)

**Description:** Takes a 'glyphSet' argument (dict), in which the glyphs that are referenced
as components are looked up by their name.

If the optional 'reverseFlipped' argument is True, components whose transformation
matrix has a negative determinant will be decomposed with a reversed path direction
to compensate for the flip.

The optional 'skipMissingComponents' argument can be set to True/False to
override the homonymous class attribute for a given pen instance.

### Function: addComponent(self, glyphName, transformation)

**Description:** Transform the points of the base glyph and draw it onto self.

### Function: addVarComponent(self, glyphName, transformation, location)

### Function: __init__(self, glyphSet)

### Function: _moveTo(self, pt)

### Function: _lineTo(self, pt)

### Function: _curveToOne(self, pt1, pt2, pt3)

### Function: _closePath(self)

### Function: _endPath(self)

### Function: _qCurveToOne(self, pt1, pt2)

**Description:** This method implements the basic quadratic curve type. The
default implementation delegates the work to the cubic curve
function. Optionally override with a native implementation.

### Function: _getCurrentPoint(self)

**Description:** Return the current point. This is not part of the public
interface, yet is useful for subclasses.

### Function: closePath(self)

### Function: endPath(self)

### Function: moveTo(self, pt)

### Function: lineTo(self, pt)

### Function: curveTo(self)

### Function: qCurveTo(self)

### Function: _moveTo(self, pt)

### Function: _lineTo(self, pt)

### Function: _curveToOne(self, bcp1, bcp2, pt)

### Function: _closePath(self)
