## AI Summary

A file named _f_p_g_m.py.


## Class: table__f_p_g_m

**Description:** Font Program table

The ``fpgm`` table typically contains function defintions that are
used by font instructions. This Font Program is similar to the Control
Value Program that is stored in the ``prep`` table, but
the ``fpgm`` table is only executed one time, when the font is first
used.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/fpgm

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __bool__(self)

**Description:** >>> fpgm = table__f_p_g_m()
>>> bool(fpgm)
False
>>> p = ttProgram.Program()
>>> fpgm.program = p
>>> bool(fpgm)
False
>>> bc = bytearray([0])
>>> p.fromBytecode(bc)
>>> bool(fpgm)
True
>>> p.bytecode.pop()
0
>>> bool(fpgm)
False
