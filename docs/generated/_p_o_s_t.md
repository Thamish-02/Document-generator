## AI Summary

A file named _p_o_s_t.py.


## Class: table__p_o_s_t

**Description:** PostScript table

The ``post`` table contains information needed to use the font on
PostScript printers, including the PostScript names of glyphs and
data that was stored in the ``FontInfo`` dictionary for Type 1 fonts.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/post

### Function: unpackPStrings(data, n)

### Function: packPStrings(strings)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: getGlyphOrder(self)

**Description:** This function will get called by a ttLib.TTFont instance.
Do not call this function yourself, use TTFont().getGlyphOrder()
or its relatives instead!

### Function: decode_format_1_0(self, data, ttFont)

### Function: decode_format_2_0(self, data, ttFont)

### Function: build_psNameMapping(self, ttFont)

### Function: decode_format_3_0(self, data, ttFont)

### Function: decode_format_4_0(self, data, ttFont)

### Function: encode_format_2_0(self, ttFont)

### Function: encode_format_4_0(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)
