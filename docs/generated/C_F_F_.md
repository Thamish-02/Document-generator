## AI Summary

A file named C_F_F_.py.


## Class: table_C_F_F_

**Description:** Compact Font Format table (version 1)

The ``CFF`` table embeds a CFF-formatted font. The CFF font format
predates OpenType and could be used as a standalone font file, but the
``CFF`` table is also used to package CFF fonts into an OpenType
container.

.. note::
   ``CFF`` has been succeeded by ``CFF2``, which eliminates much of
   the redundancy incurred by embedding CFF version 1 in an OpenType
   font.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/cff

### Function: __init__(self, tag)

### Function: decompile(self, data, otFont)

### Function: compile(self, otFont)

### Function: haveGlyphNames(self)

### Function: getGlyphOrder(self)

### Function: setGlyphOrder(self, glyphOrder)

### Function: toXML(self, writer, otFont)

### Function: fromXML(self, name, attrs, content, otFont)
