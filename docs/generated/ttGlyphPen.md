## AI Summary

A file named ttGlyphPen.py.


## Class: _TTGlyphBasePen

## Class: TTGlyphPen

**Description:** Pen used for drawing to a TrueType glyph.

This pen can be used to construct or modify glyphs in a TrueType format
font. After using the pen to draw, use the ``.glyph()`` method to retrieve
a :py:class:`~._g_l_y_f.Glyph` object representing the glyph.

## Class: TTGlyphPointPen

**Description:** Point pen used for drawing to a TrueType glyph.

This pen can be used to construct or modify glyphs in a TrueType format
font. After using the pen to draw, use the ``.glyph()`` method to retrieve
a :py:class:`~._g_l_y_f.Glyph` object representing the glyph.

### Function: __init__(self, glyphSet, handleOverflowingTransforms)

**Description:** Construct a new pen.

Args:
    glyphSet (Dict[str, Any]): A glyphset object, used to resolve components.
    handleOverflowingTransforms (bool): See below.

If ``handleOverflowingTransforms`` is True, the components' transform values
are checked that they don't overflow the limits of a F2Dot14 number:
-2.0 <= v < +2.0. If any transform value exceeds these, the composite
glyph is decomposed.

An exception to this rule is done for values that are very close to +2.0
(both for consistency with the -2.0 case, and for the relative frequency
these occur in real fonts). When almost +2.0 values occur (and all other
values are within the range -2.0 <= x <= +2.0), they are clamped to the
maximum positive value that can still be encoded as an F2Dot14: i.e.
1.99993896484375.

If False, no check is done and all components are translated unmodified
into the glyf table, followed by an inevitable ``struct.error`` once an
attempt is made to compile them.

If both contours and components are present in a glyph, the components
are decomposed.

### Function: _decompose(self, glyphName, transformation)

### Function: _isClosed(self)

**Description:** Check if the current path is closed.

### Function: init(self)

### Function: addComponent(self, baseGlyphName, transformation, identifier)

**Description:** Add a sub glyph.

### Function: _buildComponents(self, componentFlags)

### Function: glyph(self, componentFlags, dropImpliedOnCurves)

**Description:** Returns a :py:class:`~._g_l_y_f.Glyph` object representing the glyph.

Args:
    componentFlags: Flags to use for component glyphs. (default: 0x04)

    dropImpliedOnCurves: Whether to remove implied-oncurve points. (default: False)

### Function: __init__(self, glyphSet, handleOverflowingTransforms, outputImpliedClosingLine)

### Function: _addPoint(self, pt, tp)

### Function: _popPoint(self)

### Function: _isClosed(self)

### Function: lineTo(self, pt)

### Function: moveTo(self, pt)

### Function: curveTo(self)

### Function: qCurveTo(self)

### Function: closePath(self)

### Function: endPath(self)

### Function: init(self)

### Function: _isClosed(self)

### Function: beginPath(self, identifier)

**Description:** Start a new sub path.

### Function: endPath(self)

**Description:** End the current sub path.

### Function: addPoint(self, pt, segmentType, smooth, name, identifier)

**Description:** Add a point to the current sub path.
