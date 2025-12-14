## AI Summary

A file named L_T_S_H_.py.


## Class: table_L_T_S_H_

**Description:** Linear Threshold table

The ``LTSH`` table contains per-glyph settings indicating the ppem sizes
at which the advance width metric should be scaled linearly, despite the
effects of any TrueType instructions that might otherwise alter the
advance width.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/ltsh

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)
