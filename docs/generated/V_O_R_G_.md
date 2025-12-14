## AI Summary

A file named V_O_R_G_.py.


## Class: table_V_O_R_G_

**Description:** Vertical Origin table

The ``VORG`` table contains the vertical origin of each glyph
in a `CFF` or `CFF2` font.

This table is structured so that you can treat it like a dictionary keyed by glyph name.

``ttFont['VORG'][<glyphName>]`` will return the vertical origin for any glyph.

``ttFont['VORG'][<glyphName>] = <value>`` will set the vertical origin for any glyph.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/vorg

## Class: VOriginRecord

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __getitem__(self, glyphSelector)

### Function: __setitem__(self, glyphSelector, value)

### Function: __delitem__(self, glyphSelector)

### Function: __init__(self, name, vOrigin)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)
