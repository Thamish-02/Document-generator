## AI Summary

A file named ttCollection.py.


## Class: TTCollection

**Description:** Object representing a TrueType Collection / OpenType Collection.
The main API is self.fonts being a list of TTFont instances.

If shareTables is True, then different fonts in the collection
might point to the same table object if the data for the table was
the same in the font file.  Note, however, that this might result
in suprises and incorrect behavior if the different fonts involved
have different GlyphOrder.  Use only if you know what you are doing.

### Function: __init__(self, file, shareTables)

### Function: __enter__(self)

### Function: __exit__(self, type, value, traceback)

### Function: close(self)

### Function: save(self, file, shareTables)

**Description:** Save the font to disk. Similarly to the constructor,
the 'file' argument can be either a pathname or a writable
file object.

### Function: saveXML(self, fileOrPath, newlinestr, writeVersion)

### Function: __getitem__(self, item)

### Function: __setitem__(self, item, value)

### Function: __delitem__(self, item)

### Function: __len__(self)

### Function: __iter__(self)
