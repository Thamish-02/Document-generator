## AI Summary

A file named C_O_L_R_.py.


## Class: table_C_O_L_R_

**Description:** Color table

The ``COLR`` table defines color presentation of outline glyphs. It must
be used in concert with the ``CPAL`` table, which contains the color
descriptors used.

This table is structured so that you can treat it like a dictionary keyed by glyph name.

``ttFont['COLR'][<glyphName>]`` will return the color layers for any glyph.

``ttFont['COLR'][<glyphName>] = <value>`` will set the color layers for any glyph.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/colr

## Class: LayerRecord

### Function: _decompileColorLayersV0(table)

### Function: _toOTTable(self, ttFont)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __getitem__(self, glyphName)

### Function: __setitem__(self, glyphName, value)

### Function: __delitem__(self, glyphName)

### Function: __init__(self, name, colorID)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, eltname, attrs, content, ttFont)
