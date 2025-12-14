## AI Summary

A file named CFF2ToCFF.py.


### Function: _convertCFF2ToCFF(cff, otFont)

**Description:** Converts this object from CFF2 format to CFF format. This conversion
is done 'in-place'. The conversion cannot be reversed.

The CFF2 font cannot be variable. (TODO Accept those and convert to the
default instance?)

This assumes a decompiled CFF2 table. (i.e. that the object has been
filled via :meth:`decompile` and e.g. not loaded from XML.)

### Function: convertCFF2ToCFF(font)

### Function: main(args)

**Description:** Convert CFF2 OTF font to CFF OTF font
