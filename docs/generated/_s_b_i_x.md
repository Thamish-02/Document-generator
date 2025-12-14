## AI Summary

A file named _s_b_i_x.py.


## Class: table__s_b_i_x

**Description:** Standard Bitmap Graphics table

The ``sbix`` table stores bitmap image data in standard graphics formats
like JPEG, PNG, or TIFF. The glyphs for which the ``sbix`` table provides
data are indexed by Glyph ID. For each such glyph, the ``sbix`` table can
hold different data for different sizes, called "strikes."

See also https://learn.microsoft.com/en-us/typography/opentype/spec/sbix

## Class: sbixStrikeOffset

### Function: __init__(self, tag)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, xmlWriter, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)
