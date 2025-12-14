## AI Summary

A file named _c_m_a_p.py.


### Function: _make_map(font, chars, gids)

## Class: table__c_m_a_p

**Description:** Character to Glyph Index Mapping Table

This class represents the `cmap <https://docs.microsoft.com/en-us/typography/opentype/spec/cmap>`_
table, which maps between input characters (in Unicode or other system encodings)
and glyphs within the font. The ``cmap`` table contains one or more subtables
which determine the mapping of of characters to glyphs across different platforms
and encoding systems.

``table__c_m_a_p`` objects expose an accessor ``.tables`` which provides access
to the subtables, although it is normally easier to retrieve individual subtables
through the utility methods described below. To add new subtables to a font,
first determine the subtable format (if in doubt use format 4 for glyphs within
the BMP, format 12 for glyphs outside the BMP, and format 14 for Unicode Variation
Sequences) construct subtable objects with ``CmapSubtable.newSubtable(format)``,
and append them to the ``.tables`` list.

Within a subtable, the mapping of characters to glyphs is provided by the ``.cmap``
attribute.

Example::

        cmap4_0_3 = CmapSubtable.newSubtable(4)
        cmap4_0_3.platformID = 0
        cmap4_0_3.platEncID = 3
        cmap4_0_3.language = 0
        cmap4_0_3.cmap = { 0xC1: "Aacute" }

        cmap = newTable("cmap")
        cmap.tableVersion = 0
        cmap.tables = [cmap4_0_3]

See also https://learn.microsoft.com/en-us/typography/opentype/spec/cmap

## Class: CmapSubtable

**Description:** Base class for all cmap subtable formats.

Subclasses which handle the individual subtable formats are named
``cmap_format_0``, ``cmap_format_2`` etc. Use :py:meth:`getSubtableClass`
to retrieve the concrete subclass, or :py:meth:`newSubtable` to get a
new subtable object for a given format.

The object exposes a ``.cmap`` attribute, which contains a dictionary mapping
character codepoints to glyph names.

## Class: cmap_format_0

## Class: SubHeader

## Class: cmap_format_2

### Function: splitRange(startCode, endCode, cmap)

## Class: cmap_format_4

## Class: cmap_format_6

## Class: cmap_format_12_or_13

## Class: cmap_format_12

## Class: cmap_format_13

### Function: cvtToUVS(threeByteString)

### Function: cvtFromUVS(val)

## Class: cmap_format_14

## Class: cmap_format_unknown

### Function: getcmap(self, platformID, platEncID)

**Description:** Returns the first subtable which matches the given platform and encoding.

Args:
        platformID (int): The platform ID. Use 0 for Unicode, 1 for Macintosh
                (deprecated for new fonts), 2 for ISO (deprecated) and 3 for Windows.
        encodingID (int): Encoding ID. Interpretation depends on the platform ID.
                See the OpenType specification for details.

Returns:
        An object which is a subclass of :py:class:`CmapSubtable` if a matching
        subtable is found within the font, or ``None`` otherwise.

### Function: getBestCmap(self, cmapPreferences)

**Description:** Returns the 'best' Unicode cmap dictionary available in the font
or ``None``, if no Unicode cmap subtable is available.

By default it will search for the following (platformID, platEncID)
pairs in order::

                (3, 10), # Windows Unicode full repertoire
                (0, 6),  # Unicode full repertoire (format 13 subtable)
                (0, 4),  # Unicode 2.0 full repertoire
                (3, 1),  # Windows Unicode BMP
                (0, 3),  # Unicode 2.0 BMP
                (0, 2),  # Unicode ISO/IEC 10646
                (0, 1),  # Unicode 1.1
                (0, 0)   # Unicode 1.0

This particular order matches what HarfBuzz uses to choose what
subtable to use by default. This order prefers the largest-repertoire
subtable, and among those, prefers the Windows-platform over the
Unicode-platform as the former has wider support.

This order can be customized via the ``cmapPreferences`` argument.

### Function: buildReversed(self)

**Description:** Builds a reverse mapping dictionary

Iterates over all Unicode cmap tables and returns a dictionary mapping
glyphs to sets of codepoints, such as::

        {
                'one': {0x31}
                'A': {0x41,0x391}
        }

The values are sets of Unicode codepoints because
some fonts map different codepoints to the same glyph.
For example, ``U+0041 LATIN CAPITAL LETTER A`` and ``U+0391
GREEK CAPITAL LETTER ALPHA`` are sometimes the same glyph.

### Function: buildReversedMin(self)

### Function: decompile(self, data, ttFont)

### Function: ensureDecompiled(self, recurse)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: getSubtableClass(format)

**Description:** Return the subtable class for a format.

### Function: newSubtable(format)

**Description:** Return a new instance of a subtable for the given format
.

### Function: __init__(self, format)

### Function: ensureDecompiled(self, recurse)

### Function: __getattr__(self, attr)

### Function: decompileHeader(self, data, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: getEncoding(self, default)

**Description:** Returns the Python encoding name for this cmap subtable based on its platformID,
platEncID, and language.  If encoding for these values is not known, by default
``None`` is returned.  That can be overridden by passing a value to the ``default``
argument.

Note that if you want to choose a "preferred" cmap subtable, most of the time
``self.isUnicode()`` is what you want as that one only returns true for the modern,
commonly used, Unicode-compatible triplets, not the legacy ones.

### Function: isUnicode(self)

**Description:** Returns true if the characters are interpreted as Unicode codepoints.

### Function: isSymbol(self)

**Description:** Returns true if the subtable is for the Symbol encoding (3,0)

### Function: _writeCodes(self, codes, writer)

### Function: __lt__(self, other)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __init__(self)

### Function: setIDDelta(self, subHeader)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __init__(self, format)

### Function: decompileHeader(self, data, ttFont)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __init__(self, format)

### Function: _computeGIDs(self, startingGlyph, numberOfGlyphs)

### Function: _IsInSameRun(self, glyphID, lastGlyphID, charCode, lastCharCode)

### Function: __init__(self, format)

### Function: _computeGIDs(self, startingGlyph, numberOfGlyphs)

### Function: _IsInSameRun(self, glyphID, lastGlyphID, charCode, lastCharCode)

### Function: decompileHeader(self, data, ttFont)

### Function: decompile(self, data, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: decompileHeader(self, data, ttFont)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)
