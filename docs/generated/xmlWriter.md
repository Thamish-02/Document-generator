## AI Summary

A file named xmlWriter.py.


## Class: XMLWriter

### Function: escape(data)

**Description:** Escape characters not allowed in `XML 1.0 <https://www.w3.org/TR/xml/#NT-Char>`_.

### Function: escapeattr(data)

### Function: escape8bit(data)

**Description:** Input is Unicode string.

### Function: hexStr(s)

### Function: __init__(self, fileOrPath, indentwhite, idlefunc, encoding, newlinestr)

### Function: __enter__(self)

### Function: __exit__(self, exception_type, exception_value, traceback)

### Function: close(self)

### Function: write(self, string, indent)

**Description:** Writes text.

### Function: writecdata(self, string)

**Description:** Writes text in a CDATA section.

### Function: write8bit(self, data, strip)

**Description:** Writes a bytes() sequence into the XML, escaping
non-ASCII bytes.  When this is read in xmlReader,
the original bytes can be recovered by encoding to
'latin-1'.

### Function: write_noindent(self, string)

**Description:** Writes text without indentation.

### Function: _writeraw(self, data, indent, strip)

**Description:** Writes bytes, possibly indented.

### Function: newline(self)

### Function: comment(self, data)

### Function: simpletag(self, _TAG_)

### Function: begintag(self, _TAG_)

### Function: endtag(self, _TAG_)

### Function: dumphex(self, data)

### Function: indent(self)

### Function: dedent(self)

### Function: stringifyattrs(self)

### Function: escapechar(c)
