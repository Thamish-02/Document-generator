## AI Summary

A file named _l_o_c_a.py.


## Class: table__l_o_c_a

**Description:** Index to Location table

The ``loca`` table stores the offsets in the ``glyf`` table that correspond
to the descriptions of each glyph. The glyphs are references by Glyph ID.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/loca

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: set(self, locations)

### Function: toXML(self, writer, ttFont)

### Function: __getitem__(self, index)

### Function: __len__(self)
