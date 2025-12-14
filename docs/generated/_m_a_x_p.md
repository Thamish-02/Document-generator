## AI Summary

A file named _m_a_x_p.py.


## Class: table__m_a_x_p

**Description:** Maximum Profile table

The ``maxp`` table contains the memory requirements for the data in
the font.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/maxp

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: recalc(self, ttFont)

**Description:** Recalculate the font bounding box, and most other maxp values except
for the TT instructions values. Also recalculate the value of bit 1
of the flags field and the font bounding box of the 'head' table.

### Function: testrepr(self)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)
