## AI Summary

A file named sfnt.py.


## Class: SFNTReader

### Function: compress(data, level)

**Description:** Compress 'data' to Zlib format. If 'USE_ZOPFLI' variable is True,
zopfli is used instead of the zlib module.
The compression 'level' must be between 0 and 9. 1 gives best speed,
9 gives best compression (0 gives no compression at all).
The default value is a compromise between speed and compression (6).

## Class: SFNTWriter

## Class: DirectoryEntry

## Class: SFNTDirectoryEntry

## Class: WOFFDirectoryEntry

## Class: WOFFFlavorData

### Function: calcChecksum(data)

**Description:** Calculate the checksum for an arbitrary block of data.

If the data length is not a multiple of four, it assumes
it is to be padded with null byte.

        >>> print(calcChecksum(b"abcd"))
        1633837924
        >>> print(calcChecksum(b"abcdxyz"))
        3655064932

### Function: readTTCHeader(file)

### Function: writeTTCHeader(file, numFonts)

### Function: __new__(cls)

**Description:** Return an instance of the SFNTReader sub-class which is compatible
with the input file type.

### Function: __init__(self, file, checkChecksums, fontNumber)

### Function: has_key(self, tag)

### Function: keys(self)

### Function: __getitem__(self, tag)

**Description:** Fetch the raw table data.

### Function: __delitem__(self, tag)

### Function: close(self)

### Function: __getstate__(self)

### Function: __setstate__(self, state)

### Function: __new__(cls)

**Description:** Return an instance of the SFNTWriter sub-class which is compatible
with the specified 'flavor'.

### Function: __init__(self, file, numTables, sfntVersion, flavor, flavorData)

### Function: setEntry(self, tag, entry)

### Function: __setitem__(self, tag, data)

**Description:** Write raw table data to disk.

### Function: __getitem__(self, tag)

### Function: close(self)

**Description:** All tables must have been written to disk. Now write the
directory.

### Function: _calcMasterChecksum(self, directory)

### Function: writeMasterChecksum(self, directory)

### Function: reordersTables(self)

### Function: __init__(self)

### Function: fromFile(self, file)

### Function: fromString(self, str)

### Function: toString(self)

### Function: __repr__(self)

### Function: loadData(self, file)

### Function: saveData(self, file, data)

### Function: decodeData(self, rawData)

### Function: encodeData(self, data)

### Function: __init__(self)

### Function: decodeData(self, rawData)

### Function: encodeData(self, data)

### Function: __init__(self, reader)

### Function: _decompress(self, rawData)
