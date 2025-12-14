## AI Summary

A file named E_B_L_C_.py.


## Class: table_E_B_L_C_

**Description:** Embedded Bitmap Location table

The ``EBLC`` table contains the locations of monochrome or grayscale
bitmaps for glyphs. It must be used in concert with the ``EBDT`` table.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/eblc

## Class: Strike

## Class: BitmapSizeTable

## Class: SbitLineMetrics

## Class: EblcIndexSubTable

### Function: _createOffsetArrayIndexSubTableMixin(formatStringForDataType)

## Class: FixedSizeIndexSubTableMixin

## Class: eblc_index_sub_table_1

## Class: eblc_index_sub_table_2

## Class: eblc_index_sub_table_3

## Class: eblc_index_sub_table_4

## Class: eblc_index_sub_table_5

### Function: getIndexFormatClass(self, indexFormat)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __init__(self)

### Function: toXML(self, strikeIndex, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont, locator)

### Function: _getXMLMetricNames(self)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: toXML(self, name, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __init__(self, data, ttFont)

### Function: __getattr__(self, attr)

### Function: ensureDecompiled(self, recurse)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: writeMetrics(self, writer, ttFont)

### Function: readMetrics(self, name, attrs, content, ttFont)

### Function: padBitmapData(self, data)

### Function: removeSkipGlyphs(self)

## Class: OffsetArrayIndexSubTableMixin

### Function: writeMetrics(self, writer, ttFont)

### Function: readMetrics(self, name, attrs, content, ttFont)

### Function: padBitmapData(self, data)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: isValidLocation(args)

### Function: decompile(self)

### Function: compile(self, ttFont)
