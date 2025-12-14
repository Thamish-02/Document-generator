## AI Summary

A file named E_B_D_T_.py.


## Class: table_E_B_D_T_

**Description:** Embedded Bitmap Data table

The ``EBDT`` table contains monochrome or grayscale bitmap data for
glyphs. It must be used in concert with the ``EBLC`` table.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/ebdt

## Class: EbdtComponent

### Function: _data2binary(data, numBits)

### Function: _binary2data(binary)

### Function: _memoize(f)

### Function: _reverseBytes(data)

**Description:** >>> bin(ord(_reverseBytes(0b00100111)))
'0b11100100'
>>> _reverseBytes(b'\x00\xf0')
b'\x00\x0f'

### Function: _writeRawImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont)

### Function: _readRawImageData(bitmapObject, name, attrs, content, ttFont)

### Function: _writeRowImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont)

### Function: _readRowImageData(bitmapObject, name, attrs, content, ttFont)

### Function: _writeBitwiseImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont)

### Function: _readBitwiseImageData(bitmapObject, name, attrs, content, ttFont)

### Function: _writeExtFileImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont)

### Function: _readExtFileImageData(bitmapObject, name, attrs, content, ttFont)

## Class: BitmapGlyph

### Function: _createBitmapPlusMetricsMixin(metricsClass)

## Class: BitAlignedBitmapMixin

## Class: ByteAlignedBitmapMixin

## Class: ebdt_bitmap_format_1

## Class: ebdt_bitmap_format_2

## Class: ebdt_bitmap_format_5

## Class: ebdt_bitmap_format_6

## Class: ebdt_bitmap_format_7

## Class: ComponentBitmapGlyph

## Class: ebdt_bitmap_format_8

## Class: ebdt_bitmap_format_9

### Function: getImageFormatClass(self, imageFormat)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

## Class: memodict

### Function: __init__(self, data, ttFont)

### Function: __getattr__(self, attr)

### Function: ensureDecompiled(self, recurse)

### Function: getFormat(self)

### Function: toXML(self, strikeIndex, glyphName, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: writeMetrics(self, writer, ttFont)

### Function: readMetrics(self, name, attrs, content, ttFont)

### Function: writeData(self, strikeIndex, glyphName, writer, ttFont)

### Function: readData(self, name, attrs, content, ttFont)

## Class: BitmapPlusMetricsMixin

### Function: _getBitRange(self, row, bitDepth, metrics)

### Function: getRow(self, row, bitDepth, metrics, reverseBytes)

### Function: setRows(self, dataRows, bitDepth, metrics, reverseBytes)

### Function: _getByteRange(self, row, bitDepth, metrics)

### Function: getRow(self, row, bitDepth, metrics, reverseBytes)

### Function: setRows(self, dataRows, bitDepth, metrics, reverseBytes)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: toXML(self, strikeIndex, glyphName, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: decompile(self)

### Function: compile(self, ttFont)

### Function: __missing__(self, key)

### Function: writeMetrics(self, writer, ttFont)

### Function: readMetrics(self, name, attrs, content, ttFont)
