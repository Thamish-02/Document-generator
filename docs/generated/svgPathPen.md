## AI Summary

A file named svgPathPen.py.


### Function: pointToString(pt, ntos)

## Class: SVGPathPen

**Description:** Pen to draw SVG path d commands.

Args:
    glyphSet: a dictionary of drawable glyph objects keyed by name
        used to resolve component references in composite glyphs.
    ntos: a callable that takes a number and returns a string, to
        customize how numbers are formatted (default: str).

:Example:
    .. code-block::

        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 0))
        >>> pen.lineTo((1, 1))
        >>> pen.curveTo((2, 2), (3, 3), (4, 4))
        >>> pen.closePath()
        >>> pen.getCommands()
        'M0 0 1 1C2 2 3 3 4 4Z'

Note:
    Fonts have a coordinate system where Y grows up, whereas in SVG,
    Y grows down.  As such, rendering path data from this pen in
    SVG typically results in upside-down glyphs.  You can fix this
    by wrapping the data from this pen in an SVG group element with
    transform, or wrap this pen in a transform pen.  For example:
    .. code-block:: python

        spen = svgPathPen.SVGPathPen(glyphset)
        pen= TransformPen(spen , (1, 0, 0, -1, 0, 0))
        glyphset[glyphname].draw(pen)
        print(tpen.getCommands())

### Function: main(args)

**Description:** Generate per-character SVG from font and text

### Function: __init__(self, glyphSet, ntos)

### Function: _handleAnchor(self)

**Description:** >>> pen = SVGPathPen(None)
>>> pen.moveTo((0, 0))
>>> pen.moveTo((10, 10))
>>> pen._commands
['M10 10']

### Function: _moveTo(self, pt)

**Description:** >>> pen = SVGPathPen(None)
>>> pen.moveTo((0, 0))
>>> pen._commands
['M0 0']

>>> pen = SVGPathPen(None)
>>> pen.moveTo((10, 0))
>>> pen._commands
['M10 0']

>>> pen = SVGPathPen(None)
>>> pen.moveTo((0, 10))
>>> pen._commands
['M0 10']

### Function: _lineTo(self, pt)

**Description:** # duplicate point
>>> pen = SVGPathPen(None)
>>> pen.moveTo((10, 10))
>>> pen.lineTo((10, 10))
>>> pen._commands
['M10 10']

# vertical line
>>> pen = SVGPathPen(None)
>>> pen.moveTo((10, 10))
>>> pen.lineTo((10, 0))
>>> pen._commands
['M10 10', 'V0']

# horizontal line
>>> pen = SVGPathPen(None)
>>> pen.moveTo((10, 10))
>>> pen.lineTo((0, 10))
>>> pen._commands
['M10 10', 'H0']

# basic
>>> pen = SVGPathPen(None)
>>> pen.lineTo((70, 80))
>>> pen._commands
['L70 80']

# basic following a moveto
>>> pen = SVGPathPen(None)
>>> pen.moveTo((0, 0))
>>> pen.lineTo((10, 10))
>>> pen._commands
['M0 0', ' 10 10']

### Function: _curveToOne(self, pt1, pt2, pt3)

**Description:** >>> pen = SVGPathPen(None)
>>> pen.curveTo((10, 20), (30, 40), (50, 60))
>>> pen._commands
['C10 20 30 40 50 60']

### Function: _qCurveToOne(self, pt1, pt2)

**Description:** >>> pen = SVGPathPen(None)
>>> pen.qCurveTo((10, 20), (30, 40))
>>> pen._commands
['Q10 20 30 40']
>>> from fontTools.misc.roundTools import otRound
>>> pen = SVGPathPen(None, ntos=lambda v: str(otRound(v)))
>>> pen.qCurveTo((3, 3), (7, 5), (11, 4))
>>> pen._commands
['Q3 3 5 4', 'Q7 5 11 4']

### Function: _closePath(self)

**Description:** >>> pen = SVGPathPen(None)
>>> pen.closePath()
>>> pen._commands
['Z']

### Function: _endPath(self)

**Description:** >>> pen = SVGPathPen(None)
>>> pen.endPath()
>>> pen._commands
[]

### Function: getCommands(self)
