## AI Summary

A file named woff2.py.


## Class: WOFF2Reader

## Class: WOFF2Writer

### Function: getKnownTagIndex(tag)

**Description:** Return index of 'tag' in woff2KnownTags list. Return 63 if not found.

## Class: WOFF2DirectoryEntry

## Class: WOFF2LocaTable

**Description:** Same as parent class. The only difference is that it attempts to preserve
the 'indexFormat' as encoded in the WOFF2 glyf table.

## Class: WOFF2GlyfTable

**Description:** Decoder/Encoder for WOFF2 'glyf' table transform.

## Class: WOFF2HmtxTable

## Class: WOFF2FlavorData

### Function: unpackBase128(data)

**Description:** Read one to five bytes from UIntBase128-encoded input string, and return
a tuple containing the decoded integer plus any leftover data.

>>> unpackBase128(b'\x3f\x00\x00') == (63, b"\x00\x00")
True
>>> unpackBase128(b'\x8f\xff\xff\xff\x7f')[0] == 4294967295
True
>>> unpackBase128(b'\x80\x80\x3f')  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
TTLibError: UIntBase128 value must not start with leading zeros
>>> unpackBase128(b'\x8f\xff\xff\xff\xff\x7f')[0]  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
TTLibError: UIntBase128-encoded sequence is longer than 5 bytes
>>> unpackBase128(b'\x90\x80\x80\x80\x00')[0]  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
TTLibError: UIntBase128 value exceeds 2**32-1

### Function: base128Size(n)

**Description:** Return the length in bytes of a UIntBase128-encoded sequence with value n.

>>> base128Size(0)
1
>>> base128Size(24567)
3
>>> base128Size(2**32-1)
5

### Function: packBase128(n)

**Description:** Encode unsigned integer in range 0 to 2**32-1 (inclusive) to a string of
bytes using UIntBase128 variable-length encoding. Produce the shortest possible
encoding.

>>> packBase128(63) == b"\x3f"
True
>>> packBase128(2**32-1) == b'\x8f\xff\xff\xff\x7f'
True

### Function: unpack255UShort(data)

**Description:** Read one to three bytes from 255UInt16-encoded input string, and return a
tuple containing the decoded integer plus any leftover data.

>>> unpack255UShort(bytechr(252))[0]
252

Note that some numbers (e.g. 506) can have multiple encodings:
>>> unpack255UShort(struct.pack("BB", 254, 0))[0]
506
>>> unpack255UShort(struct.pack("BB", 255, 253))[0]
506
>>> unpack255UShort(struct.pack("BBB", 253, 1, 250))[0]
506

### Function: pack255UShort(value)

**Description:** Encode unsigned integer in range 0 to 65535 (inclusive) to a bytestring
using 255UInt16 variable-length encoding.

>>> pack255UShort(252) == b'\xfc'
True
>>> pack255UShort(506) == b'\xfe\x00'
True
>>> pack255UShort(762) == b'\xfd\x02\xfa'
True

### Function: compress(input_file, output_file, transform_tables)

**Description:** Compress OpenType font to WOFF2.

Args:
        input_file: a file path, file or file-like object (open in binary mode)
                containing an OpenType font (either CFF- or TrueType-flavored).
        output_file: a file path, file or file-like object where to save the
                compressed WOFF2 font.
        transform_tables: Optional[Iterable[str]]: a set of table tags for which
                to enable preprocessing transformations. By default, only 'glyf'
                and 'loca' tables are transformed. An empty set means disable all
                transformations.

### Function: decompress(input_file, output_file)

**Description:** Decompress WOFF2 font to OpenType font.

Args:
        input_file: a file path, file or file-like object (open in binary mode)
                containing a compressed WOFF2 font.
        output_file: a file path, file or file-like object where to save the
                decompressed OpenType font.

### Function: main(args)

**Description:** Compress and decompress WOFF2 fonts

### Function: __init__(self, file, checkChecksums, fontNumber)

### Function: __getitem__(self, tag)

**Description:** Fetch the raw table data. Reconstruct transformed tables.

### Function: reconstructTable(self, tag)

**Description:** Reconstruct table named 'tag' from transformed data.

### Function: _reconstructGlyf(self, data, padding)

**Description:** Return recostructed glyf table data, and set the corresponding loca's
locations. Optionally pad glyph offsets to the specified number of bytes.

### Function: _reconstructLoca(self)

**Description:** Return reconstructed loca table data.

### Function: _reconstructHmtx(self, data)

**Description:** Return reconstructed hmtx table data.

### Function: _decompileTable(self, tag)

**Description:** Decompile table data and store it inside self.ttFont.

### Function: __init__(self, file, numTables, sfntVersion, flavor, flavorData)

### Function: __setitem__(self, tag, data)

**Description:** Associate new entry named 'tag' with raw table data.

### Function: close(self)

**Description:** All tags must have been specified. Now write the table data and directory.

### Function: _normaliseGlyfAndLoca(self, padding)

**Description:** Recompile glyf and loca tables, aligning glyph offsets to multiples of
'padding' size. Update the head table's 'indexToLocFormat' accordingly while
compiling loca.

### Function: _setHeadTransformFlag(self)

**Description:** Set bit 11 of 'head' table flags to indicate that the font has undergone
a lossless modifying transform. Re-compile head table data.

### Function: _decompileTable(self, tag)

**Description:** Fetch table data, decompile it, and store it inside self.ttFont.

### Function: _compileTable(self, tag)

**Description:** Compile table and store it in its 'data' attribute.

### Function: _calcSFNTChecksumsLengthsAndOffsets(self)

**Description:** Compute the 'original' SFNT checksums, lengths and offsets for checksum
adjustment calculation. Return the total size of the uncompressed font.

### Function: _transformTables(self)

**Description:** Return transformed font data.

### Function: transformTable(self, tag)

**Description:** Return transformed table data, or None if some pre-conditions aren't
met -- in which case, the non-transformed table data will be used.

### Function: _calcMasterChecksum(self)

**Description:** Calculate checkSumAdjustment.

### Function: writeMasterChecksum(self)

**Description:** Write checkSumAdjustment to the transformBuffer.

### Function: _calcTotalSize(self)

**Description:** Calculate total size of WOFF2 font, including any meta- and/or private data.

### Function: _calcFlavorDataOffsetsAndSize(self, start)

**Description:** Calculate offsets and lengths for any meta- and/or private data.

### Function: _getVersion(self)

**Description:** Return the WOFF2 font's (majorVersion, minorVersion) tuple.

### Function: _packTableDirectory(self)

**Description:** Return WOFF2 table directory data.

### Function: _writeFlavorData(self)

**Description:** Write metadata and/or private data using appropiate padding.

### Function: reordersTables(self)

### Function: fromFile(self, file)

### Function: fromString(self, data)

### Function: toString(self)

### Function: transformVersion(self)

**Description:** Return bits 6-7 of table entry's flags, which indicate the preprocessing
transformation version number (between 0 and 3).

### Function: transformVersion(self, value)

### Function: transformed(self)

**Description:** Return True if the table has any transformation, else return False.

### Function: transformed(self, booleanValue)

### Function: __init__(self, tag)

### Function: compile(self, ttFont)

### Function: __init__(self, tag)

### Function: reconstruct(self, data, ttFont)

**Description:** Decompile transformed 'glyf' data.

### Function: transform(self, ttFont)

**Description:** Return transformed 'glyf' data

### Function: _decodeGlyph(self, glyphID)

### Function: _decodeComponents(self, glyph)

### Function: _decodeCoordinates(self, glyph)

### Function: _decodeOverlapSimpleFlag(self, glyph, glyphID)

### Function: _decodeInstructions(self, glyph)

### Function: _decodeBBox(self, glyphID, glyph)

### Function: _decodeTriplets(self, glyph)

### Function: _encodeGlyph(self, glyphID)

### Function: _encodeComponents(self, glyph)

### Function: _encodeCoordinates(self, glyph)

### Function: _encodeOverlapSimpleFlag(self, glyph, glyphID)

### Function: _encodeInstructions(self, glyph)

### Function: _encodeBBox(self, glyphID, glyph)

### Function: _encodeTriplets(self, glyph)

### Function: __init__(self, tag)

### Function: reconstruct(self, data, ttFont)

### Function: transform(self, ttFont)

### Function: __init__(self, reader, data, transformedTables)

**Description:** Data class that holds the WOFF2 header major/minor version, any
metadata or private data (as bytes strings), and the set of
table tags that have transformations applied (if reader is not None),
or will have once the WOFF2 font is compiled.

Args:
        reader: an SFNTReader (or subclass) object to read flavor data from.
        data: another WOFFFlavorData object to initialise data from.
        transformedTables: set of strings containing table tags to be transformed.

Raises:
        ImportError if the brotli module is not installed.

NOTE: The 'reader' argument, on the one hand, and the 'data' and
'transformedTables' arguments, on the other hand, are mutually exclusive.

### Function: _decompress(self, rawData)

## Class: _HelpAction

## Class: _NoGlyfTransformAction

## Class: _HmtxTransformAction

### Function: withSign(flag, baseval)

### Function: __call__(self, parser, namespace, values, option_string)

### Function: __call__(self, parser, namespace, values, option_string)

### Function: __call__(self, parser, namespace, values, option_string)
