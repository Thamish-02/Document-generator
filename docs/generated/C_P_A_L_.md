## AI Summary

A file named C_P_A_L_.py.


## Class: table_C_P_A_L_

**Description:** Color Palette table

The ``CPAL`` table contains a set of one or more color palettes. The color
records in each palette can be referenced by the ``COLR`` table to specify
the colors used in a color glyph.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/cpal

## Class: Color

### Function: __init__(self, tag)

### Function: decompile(self, data, ttFont)

### Function: _decompileUInt16Array(self, data, offset, numElements, default)

### Function: _decompileUInt32Array(self, data, offset, numElements, default)

### Function: compile(self, ttFont)

### Function: _compilePalette(self, palette)

### Function: _compileColorRecords(self)

### Function: _compilePaletteTypes(self)

### Function: _compilePaletteLabels(self)

### Function: _compilePaletteEntryLabels(self)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: hex(self)

### Function: __repr__(self)

### Function: toXML(self, writer, ttFont, index)

### Function: fromHex(cls, value)

### Function: fromRGBA(cls, red, green, blue, alpha)
