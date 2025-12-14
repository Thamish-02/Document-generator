## AI Summary

A file named _g_v_a_r.py.


## Class: table__g_v_a_r

**Description:** Glyph Variations table

The ``gvar`` table provides the per-glyph variation data that
describe how glyph outlines in the ``glyf`` table change across
the variation space that is defined for the font in the ``fvar``
table.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/gvar

### Function: compileGlyph_(dataOffsetSize, variations, pointCount, axisTags, sharedCoordIndices)

### Function: decompileGlyph_(dataOffsetSize, pointCount, sharedTuples, axisTags, data)

### Function: __init__(self, tag)

### Function: compile(self, ttFont)

### Function: compileGlyphs_(self, ttFont, axisTags, sharedCoordIndices)

### Function: decompile(self, data, ttFont)

### Function: ensureDecompiled(self, recurse)

### Function: decompileOffsets_(data, tableFormat, glyphCount)

### Function: compileOffsets_(offsets)

**Description:** Packs a list of offsets into a 'gvar' offset table.

Returns a pair (bytestring, tableFormat). Bytestring is the
packed offset table. Format indicates whether the table
uses short (tableFormat=0) or long (tableFormat=1) integers.
The returned tableFormat should get packed into the flags field
of the 'gvar' header.

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: getNumPoints_(glyph)

### Function: get_read_item()

### Function: read_item(glyphName)
