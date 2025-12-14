## AI Summary

A file named _decoders.py.


## Class: ContentDecoder

## Class: IdentityDecoder

**Description:** Handle unencoded data.

## Class: DeflateDecoder

**Description:** Handle 'deflate' decoding.

See: https://stackoverflow.com/questions/1838699

## Class: GZipDecoder

**Description:** Handle 'gzip' decoding.

See: https://stackoverflow.com/questions/1838699

## Class: BrotliDecoder

**Description:** Handle 'brotli' decoding.

Requires `pip install brotlipy`. See: https://brotlipy.readthedocs.io/
    or   `pip install brotli`. See https://github.com/google/brotli
Supports both 'brotlipy' and 'Brotli' packages since they share an import
name. The top branches are for 'brotlipy' and bottom branches for 'Brotli'

## Class: ZStandardDecoder

**Description:** Handle 'zstd' RFC 8878 decoding.

Requires `pip install zstandard`.
Can be installed as a dependency of httpx using `pip install httpx[zstd]`.

## Class: MultiDecoder

**Description:** Handle the case where multiple encodings have been applied.

## Class: ByteChunker

**Description:** Handles returning byte content in fixed-size chunks.

## Class: TextChunker

**Description:** Handles returning text content in fixed-size chunks.

## Class: TextDecoder

**Description:** Handles incrementally decoding bytes into text

## Class: LineDecoder

**Description:** Handles incrementally reading lines from text.

Has the same behaviour as the stdllib splitlines,
but handling the input iteratively.

### Function: decode(self, data)

### Function: flush(self)

### Function: decode(self, data)

### Function: flush(self)

### Function: __init__(self)

### Function: decode(self, data)

### Function: flush(self)

### Function: __init__(self)

### Function: decode(self, data)

### Function: flush(self)

### Function: __init__(self)

### Function: decode(self, data)

### Function: flush(self)

### Function: __init__(self)

### Function: decode(self, data)

### Function: flush(self)

### Function: __init__(self, children)

**Description:** 'children' should be a sequence of decoders in the order in which
each was applied.

### Function: decode(self, data)

### Function: flush(self)

### Function: __init__(self, chunk_size)

### Function: decode(self, content)

### Function: flush(self)

### Function: __init__(self, chunk_size)

### Function: decode(self, content)

### Function: flush(self)

### Function: __init__(self, encoding)

### Function: decode(self, data)

### Function: flush(self)

### Function: __init__(self)

### Function: decode(self, text)

### Function: flush(self)
