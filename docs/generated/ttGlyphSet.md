## AI Summary

A file named ttGlyphSet.py.


## Class: _TTGlyphSet

**Description:** Generic dict-like GlyphSet class that pulls metrics from hmtx and
glyph shape from TrueType or CFF.

## Class: _TTGlyphSetGlyf

## Class: _TTGlyphSetCFF

## Class: _TTGlyphSetVARC

## Class: _TTGlyph

**Description:** Glyph object that supports the Pen protocol, meaning that it has
.draw() and .drawPoints() methods that take a pen object as their only
argument. Additionally there are 'width' and 'lsb' attributes, read from
the 'hmtx' table.

If the font contains a 'vmtx' table, there will also be 'height' and 'tsb'
attributes.

## Class: _TTGlyphGlyf

## Class: _TTGlyphCFF

### Function: _evaluateCondition(condition, fvarAxes, location, instancer)

## Class: _TTGlyphVARC

### Function: _setCoordinates(glyph, coord, glyfTable)

## Class: LerpGlyphSet

**Description:** A glyphset that interpolates between two other glyphsets.

Factor is typically between 0 and 1. 0 means the first glyphset,
1 means the second glyphset, and 0.5 means the average of the
two glyphsets. Other values are possible, and can be useful to
extrapolate. Defaults to 0.5.

## Class: LerpGlyph

### Function: __init__(self, font, location, glyphsMapping)

### Function: pushLocation(self, location, reset)

### Function: pushDepth(self)

### Function: __contains__(self, glyphName)

### Function: __iter__(self)

### Function: __len__(self)

### Function: has_key(self, glyphName)

### Function: __init__(self, font, location, recalcBounds)

### Function: __getitem__(self, glyphName)

### Function: __init__(self, font, location)

### Function: __getitem__(self, glyphName)

### Function: setLocation(self, location)

### Function: pushLocation(self, location, reset)

### Function: __init__(self, font, location, glyphSet)

### Function: __getitem__(self, glyphName)

### Function: __init__(self, glyphSet, glyphName)

### Function: draw(self, pen)

**Description:** Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
how that works.

### Function: drawPoints(self, pen)

**Description:** Draw the glyph onto ``pen``. See fontTools.pens.pointPen for details
how that works.

### Function: draw(self, pen)

**Description:** Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
how that works.

### Function: drawPoints(self, pen)

**Description:** Draw the glyph onto ``pen``. See fontTools.pens.pointPen for details
how that works.

### Function: _getGlyphAndOffset(self)

### Function: _getGlyphInstance(self)

### Function: draw(self, pen)

**Description:** Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
how that works.

### Function: _draw(self, pen, isPointPen)

**Description:** Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
how that works.

### Function: draw(self, pen)

### Function: drawPoints(self, pen)

### Function: __init__(self, glyphset1, glyphset2, factor)

### Function: __getitem__(self, glyphname)

### Function: __contains__(self, glyphname)

### Function: __iter__(self)

### Function: __len__(self)

### Function: __init__(self, glyphname, glyphset)

### Function: draw(self, pen)
