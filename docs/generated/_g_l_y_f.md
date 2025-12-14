## AI Summary

A file named _g_l_y_f.py.


## Class: table__g_l_y_f

**Description:** Glyph Data table

This class represents the `glyf <https://docs.microsoft.com/en-us/typography/opentype/spec/glyf>`_
table, which contains outlines for glyphs in TrueType format. In many cases,
it is easier to access and manipulate glyph outlines through the ``GlyphSet``
object returned from :py:meth:`fontTools.ttLib.ttFont.getGlyphSet`::

                >> from fontTools.pens.boundsPen import BoundsPen
                >> glyphset = font.getGlyphSet()
                >> bp = BoundsPen(glyphset)
                >> glyphset["A"].draw(bp)
                >> bp.bounds
                (19, 0, 633, 716)

However, this class can be used for low-level access to the ``glyf`` table data.
Objects of this class support dictionary-like access, mapping glyph names to
:py:class:`Glyph` objects::

                >> glyf = font["glyf"]
                >> len(glyf["Aacute"].components)
                2

Note that when adding glyphs to the font via low-level access to the ``glyf``
table, the new glyphs must also be added to the ``hmtx``/``vmtx`` table::

                >> font["glyf"]["divisionslash"] = Glyph()
                >> font["hmtx"]["divisionslash"] = (640, 0)

### Function: flagBest(x, y, onCurve)

**Description:** For a given x,y delta pair, returns the flag that packs this pair
most efficiently, as well as the number of byte cost of such flag.

### Function: flagFits(newFlag, oldFlag, mask)

### Function: flagSupports(newFlag, oldFlag)

### Function: flagEncodeCoord(flag, mask, coord, coordBytes)

### Function: flagEncodeCoords(flag, x, y, xBytes, yBytes)

## Class: Glyph

**Description:** This class represents an individual TrueType glyph.

TrueType glyph objects come in two flavours: simple and composite. Simple
glyph objects contain contours, represented via the ``.coordinates``,
``.flags``, ``.numberOfContours``, and ``.endPtsOfContours`` attributes;
composite glyphs contain components, available through the ``.components``
attributes.

Because the ``.coordinates`` attribute (and other simple glyph attributes mentioned
above) is only set on simple glyphs and the ``.components`` attribute is only
set on composite glyphs, it is necessary to use the :py:meth:`isComposite`
method to test whether a glyph is simple or composite before attempting to
access its data.

For a composite glyph, the components can also be accessed via array-like access::

        >> assert(font["glyf"]["Aacute"].isComposite())
        >> font["glyf"]["Aacute"][0]
        <fontTools.ttLib.tables._g_l_y_f.GlyphComponent at 0x1027b2ee0>

### Function: _is_mid_point(p0, p1, p2)

### Function: dropImpliedOnCurvePoints()

**Description:** Drop impliable on-curve points from the (simple) glyph or glyphs.

In TrueType glyf outlines, on-curve points can be implied when they are located at
the midpoint of the line connecting two consecutive off-curve points.

If more than one glyphs are passed, these are assumed to be interpolatable masters
of the same glyph impliable, and thus only the on-curve points that are impliable
for all of them will actually be implied.
Composite glyphs or empty glyphs are skipped, only simple glyphs with 1 or more
contours are considered.
The input glyph(s) is/are modified in-place.

Args:
    interpolatable_glyphs: The glyph or glyphs to modify in-place.

Returns:
    The set of point indices that were dropped if any.

Raises:
    ValueError if simple glyphs are not in fact interpolatable because they have
    different point flags or number of contours.

Reference:
https://developer.apple.com/fonts/TrueType-Reference-Manual/RM01/Chap1.html

## Class: GlyphComponent

**Description:** Represents a component within a composite glyph.

The component is represented internally with four attributes: ``glyphName``,
``x``, ``y`` and ``transform``. If there is no "two-by-two" matrix (i.e
no scaling, reflection, or rotation; only translation), the ``transform``
attribute is not present.

## Class: GlyphCoordinates

**Description:** A list of glyph coordinates.

Unlike an ordinary list, this is a numpy-like matrix object which supports
matrix addition, scalar multiplication and other operations described below.

### Function: decompile(self, data, ttFont)

### Function: ensureDecompiled(self, recurse)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont, splitGlyphs)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: setGlyphOrder(self, glyphOrder)

**Description:** Sets the glyph order

Args:
        glyphOrder ([str]): List of glyph names in order.

### Function: getGlyphName(self, glyphID)

**Description:** Returns the name for the glyph with the given ID.

Raises a ``KeyError`` if the glyph name is not found in the font.

### Function: _buildReverseGlyphOrderDict(self)

### Function: getGlyphID(self, glyphName)

**Description:** Returns the ID of the glyph with the given name.

Raises a ``ValueError`` if the glyph is not found in the font.

### Function: removeHinting(self)

**Description:** Removes TrueType hints from all glyphs in the glyphset.

See :py:meth:`Glyph.removeHinting`.

### Function: keys(self)

### Function: has_key(self, glyphName)

### Function: get(self, glyphName, default)

### Function: __getitem__(self, glyphName)

### Function: __setitem__(self, glyphName, glyph)

### Function: __delitem__(self, glyphName)

### Function: __len__(self)

### Function: _getPhantomPoints(self, glyphName, hMetrics, vMetrics)

**Description:** Compute the four "phantom points" for the given glyph from its bounding box
and the horizontal and vertical advance widths and sidebearings stored in the
ttFont's "hmtx" and "vmtx" tables.

'hMetrics' should be ttFont['hmtx'].metrics.

'vMetrics' should be ttFont['vmtx'].metrics if there is "vmtx" or None otherwise.
If there is no vMetrics passed in, vertical phantom points are set to the zero coordinate.

https://docs.microsoft.com/en-us/typography/opentype/spec/tt_instructing_glyphs#phantoms

### Function: _getCoordinatesAndControls(self, glyphName, hMetrics, vMetrics)

**Description:** Return glyph coordinates and controls as expected by "gvar" table.

The coordinates includes four "phantom points" for the glyph metrics,
as mandated by the "gvar" spec.

The glyph controls is a namedtuple with the following attributes:
        - numberOfContours: -1 for composite glyphs.
        - endPts: list of indices of end points for each contour in simple
        glyphs, or component indices in composite glyphs (used for IUP
        optimization).
        - flags: array of contour point flags for simple glyphs (None for
        composite glyphs).
        - components: list of base glyph names (str) for each component in
        composite glyphs (None for simple glyphs).

The "hMetrics" and vMetrics are used to compute the "phantom points" (see
the "_getPhantomPoints" method).

Return None if the requested glyphName is not present.

### Function: _setCoordinates(self, glyphName, coord, hMetrics, vMetrics)

**Description:** Set coordinates and metrics for the given glyph.

"coord" is an array of GlyphCoordinates which must include the "phantom
points" as the last four coordinates.

Both the horizontal/vertical advances and left/top sidebearings in "hmtx"
and "vmtx" tables (if any) are updated from four phantom points and
the glyph's bounding boxes.

The "hMetrics" and vMetrics are used to propagate "phantom points"
into "hmtx" and "vmtx" tables if desired.  (see the "_getPhantomPoints"
method).

### Function: _synthesizeVMetrics(self, glyphName, ttFont, defaultVerticalOrigin)

**Description:** This method is wrong and deprecated.
For rationale see:
https://github.com/fonttools/fonttools/pull/2266/files#r613569473

### Function: getPhantomPoints(self, glyphName, ttFont, defaultVerticalOrigin)

**Description:** Old public name for self._getPhantomPoints().
See: https://github.com/fonttools/fonttools/pull/2266

### Function: getCoordinatesAndControls(self, glyphName, ttFont, defaultVerticalOrigin)

**Description:** Old public name for self._getCoordinatesAndControls().
See: https://github.com/fonttools/fonttools/pull/2266

### Function: setCoordinates(self, glyphName, ttFont)

**Description:** Old public name for self._setCoordinates().
See: https://github.com/fonttools/fonttools/pull/2266

### Function: __init__(self, data)

### Function: compact(self, glyfTable, recalcBBoxes)

### Function: expand(self, glyfTable)

### Function: compile(self, glyfTable, recalcBBoxes)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: getCompositeMaxpValues(self, glyfTable, maxComponentDepth)

### Function: getMaxpValues(self)

### Function: decompileComponents(self, data, glyfTable)

### Function: decompileCoordinates(self, data)

### Function: decompileCoordinatesRaw(self, nCoordinates, data, pos)

### Function: compileComponents(self, glyfTable)

### Function: compileCoordinates(self)

### Function: compileDeltasGreedy(self, flags, deltas)

### Function: compileDeltasOptimal(self, flags, deltas)

### Function: compileDeltasForSpeed(self, flags, deltas)

### Function: recalcBounds(self, glyfTable)

**Description:** Recalculates the bounds of the glyph.

Each glyph object stores its bounding box in the
``xMin``/``yMin``/``xMax``/``yMax`` attributes. These bounds must be
recomputed when the ``coordinates`` change. The ``table__g_l_y_f`` bounds
must be provided to resolve component bounds.

### Function: tryRecalcBoundsComposite(self, glyfTable)

**Description:** Try recalculating the bounds of a composite glyph that has
certain constrained properties. Namely, none of the components
have a transform other than an integer translate, and none
uses the anchor points.

Each glyph object stores its bounding box in the
``xMin``/``yMin``/``xMax``/``yMax`` attributes. These bounds must be
recomputed when the ``coordinates`` change. The ``table__g_l_y_f`` bounds
must be provided to resolve component bounds.

Return True if bounds were calculated, False otherwise.

### Function: isComposite(self)

**Description:** Test whether a glyph has components

### Function: getCoordinates(self, glyfTable)

**Description:** Return the coordinates, end points and flags

This method returns three values: A :py:class:`GlyphCoordinates` object,
a list of the indexes of the final points of each contour (allowing you
to split up the coordinates list into contours) and a list of flags.

On simple glyphs, this method returns information from the glyph's own
contours; on composite glyphs, it "flattens" all components recursively
to return a list of coordinates representing all the components involved
in the glyph.

To interpret the flags for each point, see the "Simple Glyph Flags"
section of the `glyf table specification <https://docs.microsoft.com/en-us/typography/opentype/spec/glyf#simple-glyph-description>`.

### Function: getComponentNames(self, glyfTable)

**Description:** Returns a list of names of component glyphs used in this glyph

This method can be used on simple glyphs (in which case it returns an
empty list) or composite glyphs.

### Function: trim(self, remove_hinting)

**Description:** Remove padding and, if requested, hinting, from a glyph.
This works on both expanded and compacted glyphs, without
expanding it.

### Function: removeHinting(self)

**Description:** Removes TrueType hinting instructions from the glyph.

### Function: draw(self, pen, glyfTable, offset)

**Description:** Draws the glyph using the supplied pen object.

Arguments:
        pen: An object conforming to the pen protocol.
        glyfTable: A :py:class:`table__g_l_y_f` object, to resolve components.
        offset (int): A horizontal offset. If provided, all coordinates are
                translated by this offset.

### Function: drawPoints(self, pen, glyfTable, offset)

**Description:** Draw the glyph using the supplied pointPen. As opposed to Glyph.draw(),
this will not change the point indices.

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __init__(self)

### Function: getComponentInfo(self)

**Description:** Return information about the component

This method returns a tuple of two values: the glyph name of the component's
base glyph, and a transformation matrix. As opposed to accessing the attributes
directly, ``getComponentInfo`` always returns a six-element tuple of the
component's transformation matrix, even when the two-by-two ``.transform``
matrix is not present.

### Function: decompile(self, data, glyfTable)

### Function: compile(self, more, haveInstructions, glyfTable)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: _hasOnlyIntegerTranslate(self)

**Description:** Return True if it's a 'simple' component.

That is, it has no anchor points and no transform other than integer translate.

### Function: __init__(self, iterable)

### Function: array(self)

**Description:** Returns the underlying array of coordinates

### Function: zeros(count)

**Description:** Creates a new ``GlyphCoordinates`` object with all coordinates set to (0,0)

### Function: copy(self)

**Description:** Creates a new ``GlyphCoordinates`` object which is a copy of the current one.

### Function: __len__(self)

**Description:** Returns the number of coordinates in the array.

### Function: __getitem__(self, k)

**Description:** Returns a two element tuple (x,y)

### Function: __setitem__(self, k, v)

**Description:** Sets a point's coordinates to a two element tuple (x,y)

### Function: __delitem__(self, i)

**Description:** Removes a point from the list

### Function: __repr__(self)

### Function: append(self, p)

### Function: extend(self, iterable)

### Function: toInt(self)

### Function: calcBounds(self)

### Function: calcIntBounds(self, round)

### Function: relativeToAbsolute(self)

### Function: absoluteToRelative(self)

### Function: translate(self, p)

**Description:** >>> GlyphCoordinates([(1,2)]).translate((.5,0))

### Function: scale(self, p)

**Description:** >>> GlyphCoordinates([(1,2)]).scale((.5,0))

### Function: transform(self, t)

**Description:** >>> GlyphCoordinates([(1,2)]).transform(((.5,0),(.2,.5)))

### Function: __eq__(self, other)

**Description:** >>> g = GlyphCoordinates([(1,2)])
>>> g2 = GlyphCoordinates([(1.0,2)])
>>> g3 = GlyphCoordinates([(1.5,2)])
>>> g == g2
True
>>> g == g3
False
>>> g2 == g3
False

### Function: __ne__(self, other)

**Description:** >>> g = GlyphCoordinates([(1,2)])
>>> g2 = GlyphCoordinates([(1.0,2)])
>>> g3 = GlyphCoordinates([(1.5,2)])
>>> g != g2
False
>>> g != g3
True
>>> g2 != g3
True

### Function: __pos__(self)

**Description:** >>> g = GlyphCoordinates([(1,2)])
>>> g
GlyphCoordinates([(1, 2)])
>>> g2 = +g
>>> g2
GlyphCoordinates([(1, 2)])
>>> g2.translate((1,0))
>>> g2
GlyphCoordinates([(2, 2)])
>>> g
GlyphCoordinates([(1, 2)])

### Function: __neg__(self)

**Description:** >>> g = GlyphCoordinates([(1,2)])
>>> g
GlyphCoordinates([(1, 2)])
>>> g2 = -g
>>> g2
GlyphCoordinates([(-1, -2)])
>>> g
GlyphCoordinates([(1, 2)])

### Function: __round__(self)

### Function: __add__(self, other)

### Function: __sub__(self, other)

### Function: __mul__(self, other)

### Function: __truediv__(self, other)

### Function: __rsub__(self, other)

### Function: __iadd__(self, other)

**Description:** >>> g = GlyphCoordinates([(1,2)])
>>> g += (.5,0)
>>> g
GlyphCoordinates([(1.5, 2)])
>>> g2 = GlyphCoordinates([(3,4)])
>>> g += g2
>>> g
GlyphCoordinates([(4.5, 6)])

### Function: __isub__(self, other)

**Description:** >>> g = GlyphCoordinates([(1,2)])
>>> g -= (.5,0)
>>> g
GlyphCoordinates([(0.5, 2)])
>>> g2 = GlyphCoordinates([(3,4)])
>>> g -= g2
>>> g
GlyphCoordinates([(-2.5, -2)])

### Function: __imul__(self, other)

**Description:** >>> g = GlyphCoordinates([(1,2)])
>>> g *= (2,.5)
>>> g *= 2
>>> g
GlyphCoordinates([(4, 2)])
>>> g = GlyphCoordinates([(1,2)])
>>> g *= 2
>>> g
GlyphCoordinates([(2, 4)])

### Function: __itruediv__(self, other)

**Description:** >>> g = GlyphCoordinates([(1,3)])
>>> g /= (.5,1.5)
>>> g /= 2
>>> g
GlyphCoordinates([(1, 1)])

### Function: __bool__(self)

**Description:** >>> g = GlyphCoordinates([])
>>> bool(g)
False
>>> g = GlyphCoordinates([(0,0), (0.,0)])
>>> bool(g)
True
>>> g = GlyphCoordinates([(0,0), (1,0)])
>>> bool(g)
True
>>> g = GlyphCoordinates([(0,.5), (0,0)])
>>> bool(g)
True
