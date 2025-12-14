## AI Summary

A file named agl.py.


## Class: AGLError

### Function: _builddicts()

### Function: toUnicode(glyph, isZapfDingbats)

**Description:** Convert glyph names to Unicode, such as ``'longs_t.oldstyle'`` --> ``u'ſt'``

If ``isZapfDingbats`` is ``True``, the implementation recognizes additional
glyph names (as required by the AGL specification).

### Function: _glyphComponentToUnicode(component, isZapfDingbats)

### Function: _zapfDingbatsToUnicode(glyph)

**Description:** Helper for toUnicode().

### Function: _uniToUnicode(component)

**Description:** Helper for toUnicode() to handle "uniABCD" components.

### Function: _uToUnicode(component)

**Description:** Helper for toUnicode() to handle "u1ABCD" components.
