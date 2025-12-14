## AI Summary

A file named V_D_M_X_.py.


## Class: table_V_D_M_X_

**Description:** Vertical Device Metrics table

The ``VDMX`` table records changes to the vertical glyph minima
and maxima that result from Truetype instructions.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/vdmx

### Function: decompile(self, data, ttFont)

### Function: _getOffsets(self)

**Description:** Calculate offsets to VDMX_Group records.
For each ratRange return a list of offset values from the beginning of
the VDMX table to a VDMX_Group.

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)
