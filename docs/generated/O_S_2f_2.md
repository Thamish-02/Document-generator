## AI Summary

A file named O_S_2f_2.py.


## Class: Panose

## Class: table_O_S_2f_2

**Description:** OS/2 and Windows Metrics table

The ``OS/2`` table contains a variety of font-wide metrics and
parameters that may be useful to an operating system or other
software for system-integration purposes.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/os2

### Function: _getUnicodeRanges()

### Function: intersectUnicodeRanges(unicodes, inverse)

**Description:** Intersect a sequence of (int) Unicode codepoints with the Unicode block
ranges defined in the OpenType specification v1.7, and return the set of
'ulUnicodeRanges' bits for which there is at least ONE intersection.
If 'inverse' is True, return the the bits for which there is NO intersection.

>>> intersectUnicodeRanges([0x0410]) == {9}
True
>>> intersectUnicodeRanges([0x0410, 0x1F000]) == {9, 57, 122}
True
>>> intersectUnicodeRanges([0x0410, 0x1F000], inverse=True) == (
...     set(range(len(OS2_UNICODE_RANGES))) - {9, 57, 122})
True

### Function: calcCodePageRanges(unicodes)

**Description:** Given a set of Unicode codepoints (integers), calculate the
corresponding OS/2 CodePage range bits.
This is a direct translation of FontForge implementation:
https://github.com/fontforge/fontforge/blob/7b2c074/fontforge/tottf.c#L3158

### Function: __init__(self)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: updateFirstAndLastCharIndex(self, ttFont)

### Function: usMaxContex(self)

### Function: usMaxContex(self, value)

### Function: fsFirstCharIndex(self)

### Function: fsFirstCharIndex(self, value)

### Function: fsLastCharIndex(self)

### Function: fsLastCharIndex(self, value)

### Function: getUnicodeRanges(self)

**Description:** Return the set of 'ulUnicodeRange*' bits currently enabled.

### Function: setUnicodeRanges(self, bits)

**Description:** Set the 'ulUnicodeRange*' fields to the specified 'bits'.

### Function: recalcUnicodeRanges(self, ttFont, pruneOnly)

**Description:** Intersect the codepoints in the font's Unicode cmap subtables with
the Unicode block ranges defined in the OpenType specification (v1.7),
and set the respective 'ulUnicodeRange*' bits if there is at least ONE
intersection.
If 'pruneOnly' is True, only clear unused bits with NO intersection.

### Function: getCodePageRanges(self)

**Description:** Return the set of 'ulCodePageRange*' bits currently enabled.

### Function: setCodePageRanges(self, bits)

**Description:** Set the 'ulCodePageRange*' fields to the specified 'bits'.

### Function: recalcCodePageRanges(self, ttFont, pruneOnly)

### Function: recalcAvgCharWidth(self, ttFont)

**Description:** Recalculate xAvgCharWidth using metrics from ttFont's 'hmtx' table.

Set it to 0 if the unlikely event 'hmtx' table is not found.
