## AI Summary

A file named _inputstream.py.


## Class: BufferedStream

**Description:** Buffering for streams that do not have buffering of their own

The buffer is implemented as a list of chunks on the assumption that
joining many strings will be slow since it is O(n**2)

### Function: HTMLInputStream(source)

## Class: HTMLUnicodeInputStream

**Description:** Provides a unicode stream of characters to the HTMLTokenizer.

This class takes care of character encoding and removing or replacing
incorrect byte-sequences and also provides column and line tracking.

## Class: HTMLBinaryInputStream

**Description:** Provides a unicode stream of characters to the HTMLTokenizer.

This class takes care of character encoding and removing or replacing
incorrect byte-sequences and also provides column and line tracking.

## Class: EncodingBytes

**Description:** String-like object with an associated position and various extra methods
If the position is ever greater than the string length then an exception is
raised

## Class: EncodingParser

**Description:** Mini parser for detecting character encoding from meta elements

## Class: ContentAttrParser

### Function: lookupEncoding(encoding)

**Description:** Return the python codec name corresponding to an encoding or None if the
string doesn't correspond to a valid encoding.

### Function: __init__(self, stream)

### Function: tell(self)

### Function: seek(self, pos)

### Function: read(self, bytes)

### Function: _bufferedBytes(self)

### Function: _readStream(self, bytes)

### Function: _readFromBuffer(self, bytes)

### Function: __init__(self, source)

**Description:** Initialises the HTMLInputStream.

HTMLInputStream(source, [encoding]) -> Normalized stream from source
for use by html5lib.

source can be either a file-object, local filename or a string.

The optional encoding parameter must be a string that indicates
the encoding.  If specified, that encoding will be used,
regardless of any BOM or later declaration (such as in a meta
element)

### Function: reset(self)

### Function: openStream(self, source)

**Description:** Produces a file object from source.

source can be either a file object, local filename or a string.

### Function: _position(self, offset)

### Function: position(self)

**Description:** Returns (line, col) of the current position in the stream.

### Function: char(self)

**Description:** Read one character from the stream or queue if available. Return
EOF when EOF is reached.

### Function: readChunk(self, chunkSize)

### Function: characterErrorsUCS4(self, data)

### Function: characterErrorsUCS2(self, data)

### Function: charsUntil(self, characters, opposite)

**Description:** Returns a string of characters from the stream up to but not
including any character in 'characters' or EOF. 'characters' must be
a container that supports the 'in' method and iteration over its
characters.

### Function: unget(self, char)

### Function: __init__(self, source, override_encoding, transport_encoding, same_origin_parent_encoding, likely_encoding, default_encoding, useChardet)

**Description:** Initialises the HTMLInputStream.

HTMLInputStream(source, [encoding]) -> Normalized stream from source
for use by html5lib.

source can be either a file-object, local filename or a string.

The optional encoding parameter must be a string that indicates
the encoding.  If specified, that encoding will be used,
regardless of any BOM or later declaration (such as in a meta
element)

### Function: reset(self)

### Function: openStream(self, source)

**Description:** Produces a file object from source.

source can be either a file object, local filename or a string.

### Function: determineEncoding(self, chardet)

### Function: changeEncoding(self, newEncoding)

### Function: detectBOM(self)

**Description:** Attempts to detect at BOM at the start of the stream. If
an encoding can be determined from the BOM return the name of the
encoding otherwise return None

### Function: detectEncodingMeta(self)

**Description:** Report the encoding declared by the meta element
        

### Function: __new__(self, value)

### Function: __init__(self, value)

### Function: __iter__(self)

### Function: __next__(self)

### Function: next(self)

### Function: previous(self)

### Function: setPosition(self, position)

### Function: getPosition(self)

### Function: getCurrentByte(self)

### Function: skip(self, chars)

**Description:** Skip past a list of characters

### Function: skipUntil(self, chars)

### Function: matchBytes(self, bytes)

**Description:** Look for a sequence of bytes at the start of a string. If the bytes
are found return True and advance the position to the byte after the
match. Otherwise return False and leave the position alone

### Function: jumpTo(self, bytes)

**Description:** Look for the next sequence of bytes matching a given sequence. If
a match is found advance the position to the last byte of the match

### Function: __init__(self, data)

**Description:** string - the data to work on for encoding detection

### Function: getEncoding(self)

### Function: handleComment(self)

**Description:** Skip over comments

### Function: handleMeta(self)

### Function: handlePossibleStartTag(self)

### Function: handlePossibleEndTag(self)

### Function: handlePossibleTag(self, endTag)

### Function: handleOther(self)

### Function: getAttribute(self)

**Description:** Return a name,value pair for the next attribute in the stream,
if one is found, or None

### Function: __init__(self, data)

### Function: parse(self)
