## AI Summary

A file named statisticsPen.py.


## Class: StatisticsBase

## Class: StatisticsPen

**Description:** Pen calculating area, center of mass, variance and
standard-deviation, covariance and correlation, and slant,
of glyph shapes.

Note that if the glyph shape is self-intersecting, the values
are not correct (but well-defined). Moreover, area will be
negative if contour directions are clockwise.

## Class: StatisticsControlPen

**Description:** Pen calculating area, center of mass, variance and
standard-deviation, covariance and correlation, and slant,
of glyph shapes, using the control polygon only.

Note that if the glyph shape is self-intersecting, the values
are not correct (but well-defined). Moreover, area will be
negative if contour directions are clockwise.

### Function: _test(glyphset, upem, glyphs, quiet)

### Function: main(args)

**Description:** Report font glyph shape geometricsl statistics

### Function: __init__(self)

### Function: _zero(self)

### Function: _update(self)

### Function: __init__(self, glyphset)

### Function: _closePath(self)

### Function: _update(self)

### Function: __init__(self, glyphset)

### Function: _moveTo(self, pt)

### Function: _lineTo(self, pt)

### Function: _qCurveToOne(self, pt1, pt2)

### Function: _curveToOne(self, pt1, pt2, pt3)

### Function: _closePath(self)

### Function: _endPath(self)

### Function: _update(self)
