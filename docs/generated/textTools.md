## AI Summary

A file named textTools.py.


## Class: Tag

### Function: readHex(content)

**Description:** Convert a list of hex strings to binary data.

### Function: deHexStr(hexdata)

**Description:** Convert a hex string to binary data.

### Function: hexStr(data)

**Description:** Convert binary data to a hex string.

### Function: num2binary(l, bits)

### Function: binary2num(bin)

### Function: caselessSort(alist)

**Description:** Return a sorted copy of a list. If there are only strings
in the list, it will not consider case.

### Function: pad(data, size)

**Description:** Pad byte string 'data' with null bytes until its length is a
multiple of 'size'.

>>> len(pad(b'abcd', 4))
4
>>> len(pad(b'abcde', 2))
6
>>> len(pad(b'abcde', 4))
8
>>> pad(b'abcdef', 4) == b'abcdef\x00\x00'
True

### Function: tostr(s, encoding, errors)

### Function: tobytes(s, encoding, errors)

### Function: bytechr(n)

### Function: byteord(c)

### Function: strjoin(iterable, joiner)

### Function: bytesjoin(iterable, joiner)

### Function: transcode(blob)

### Function: __new__(self, content)

### Function: __ne__(self, other)

### Function: __eq__(self, other)

### Function: __hash__(self)

### Function: tobytes(self)
