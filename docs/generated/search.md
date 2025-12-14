## AI Summary

A file named search.py.


## Class: Pattern

**Description:** Base class for search patterns.

The following L{Pattern} subclasses are provided by WinAppDbg:
 - L{BytePattern}
 - L{TextPattern}
 - L{RegExpPattern}
 - L{HexPattern}

@see: L{Search.search_process}

## Class: BytePattern

**Description:** Fixed byte pattern.

@type pattern: str
@ivar pattern: Byte string to search for.

@type length: int
@ivar length: Length of the byte pattern.

## Class: TextPattern

**Description:** Text pattern.

@type isUnicode: bool
@ivar isUnicode: C{True} if the text to search for is a compat.unicode string,
    C{False} otherwise.

@type encoding: str
@ivar encoding: Encoding for the text parameter.
    Only used when the text to search for is a Unicode string.
    Don't change unless you know what you're doing!

@type caseSensitive: bool
@ivar caseSensitive: C{True} of the search is case sensitive,
    C{False} otherwise.

## Class: RegExpPattern

**Description:** Regular expression pattern.

@type pattern: str
@ivar pattern: Regular expression in text form.

@type flags: int
@ivar flags: Regular expression flags.

@type regexp: re.compile
@ivar regexp: Regular expression in compiled form.

@type maxLength: int
@ivar maxLength:
    Maximum expected length of the strings matched by this regular
    expression.

    This value will be used to calculate the required buffer size when
    doing buffered searches.

    Ideally it should be an exact value, but in some cases it's not
    possible to calculate so an upper limit should be given instead.

    If that's not possible either, C{None} should be used. That will
    cause an exception to be raised if this pattern is used in a
    buffered search.

## Class: HexPattern

**Description:** Hexadecimal pattern.

Hex patterns must be in this form::
    "68 65 6c 6c 6f 20 77 6f 72 6c 64"  # "hello world"

Spaces are optional. Capitalization of hex digits doesn't matter.
This is exactly equivalent to the previous example::
    "68656C6C6F20776F726C64"            # "hello world"

Wildcards are allowed, in the form of a C{?} sign in any hex digit::
    "5? 5? c3"          # pop register / pop register / ret
    "b8 ?? ?? ?? ??"    # mov eax, immediate value

@type pattern: str
@ivar pattern: Hexadecimal pattern.

## Class: Search

**Description:** Static class to group the search functionality.

Do not instance this class! Use its static methods instead.

### Function: __init__(self, pattern)

**Description:** Class constructor.

The only mandatory argument should be the pattern string.

This method B{MUST} be reimplemented by subclasses of L{Pattern}.

### Function: __len__(self)

**Description:** Returns the maximum expected length of the strings matched by this
pattern. Exact behavior is implementation dependent.

Ideally it should be an exact value, but in some cases it's not
possible to calculate so an upper limit should be returned instead.

If that's not possible either an exception must be raised.

This value will be used to calculate the required buffer size when
doing buffered searches.

This method B{MUST} be reimplemented by subclasses of L{Pattern}.

### Function: read(self, process, address, size)

**Description:** Reads the requested number of bytes from the process memory at the
given address.

Subclasses of L{Pattern} tipically don't need to reimplement this
method.

### Function: find(self, buffer, pos)

**Description:** Searches for the pattern in the given buffer, optionally starting at
the given position within the buffer.

This method B{MUST} be reimplemented by subclasses of L{Pattern}.

@type  buffer: str
@param buffer: Buffer to search on.

@type  pos: int
@param pos:
    (Optional) Position within the buffer to start searching from.

@rtype:  tuple( int, int )
@return: Tuple containing the following:
     - Position within the buffer where a match is found, or C{-1} if
       no match was found.
     - Length of the matched data if a match is found, or undefined if
       no match was found.

### Function: found(self, address, size, data)

**Description:** This method gets called when a match is found.

This allows subclasses of L{Pattern} to filter out unwanted results,
or modify the results before giving them to the caller of
L{Search.search_process}.

If the return value is C{None} the result is skipped.

Subclasses of L{Pattern} don't need to reimplement this method unless
filtering is needed.

@type  address: int
@param address: The memory address where the pattern was found.

@type  size: int
@param size: The size of the data that matches the pattern.

@type  data: str
@param data: The data that matches the pattern.

@rtype:  tuple( int, int, str )
@return: Tuple containing the following:
     * The memory address where the pattern was found.
     * The size of the data that matches the pattern.
     * The data that matches the pattern.

### Function: __init__(self, pattern)

**Description:** @type  pattern: str
@param pattern: Byte string to search for.

### Function: __len__(self)

**Description:** Returns the exact length of the pattern.

@see: L{Pattern.__len__}

### Function: find(self, buffer, pos)

### Function: __init__(self, text, encoding, caseSensitive)

**Description:** @type  text: str or compat.unicode
@param text: Text to search for.

@type  encoding: str
@param encoding: (Optional) Encoding for the text parameter.
    Only used when the text to search for is a Unicode string.
    Don't change unless you know what you're doing!

@type  caseSensitive: bool
@param caseSensitive: C{True} of the search is case sensitive,
    C{False} otherwise.

### Function: read(self, process, address, size)

### Function: found(self, address, size, data)

### Function: __init__(self, regexp, flags, maxLength)

**Description:** @type  regexp: str
@param regexp: Regular expression string.

@type  flags: int
@param flags: Regular expression flags.

@type  maxLength: int
@param maxLength: Maximum expected length of the strings matched by
    this regular expression.

    This value will be used to calculate the required buffer size when
    doing buffered searches.

    Ideally it should be an exact value, but in some cases it's not
    possible to calculate so an upper limit should be given instead.

    If that's not possible either, C{None} should be used. That will
    cause an exception to be raised if this pattern is used in a
    buffered search.

### Function: __len__(self)

**Description:** Returns the maximum expected length of the strings matched by this
pattern. This value is taken from the C{maxLength} argument of the
constructor if this class.

Ideally it should be an exact value, but in some cases it's not
possible to calculate so an upper limit should be returned instead.

If that's not possible either an exception must be raised.

This value will be used to calculate the required buffer size when
doing buffered searches.

### Function: find(self, buffer, pos)

### Function: __new__(cls, pattern)

**Description:** If the pattern is completely static (no wildcards are present) a
L{BytePattern} is created instead. That's because searching for a
fixed byte pattern is faster than searching for a regular expression.

### Function: __init__(self, hexa)

**Description:** Hex patterns must be in this form::
    "68 65 6c 6c 6f 20 77 6f 72 6c 64"  # "hello world"

Spaces are optional. Capitalization of hex digits doesn't matter.
This is exactly equivalent to the previous example::
    "68656C6C6F20776F726C64"            # "hello world"

Wildcards are allowed, in the form of a C{?} sign in any hex digit::
    "5? 5? c3"          # pop register / pop register / ret
    "b8 ?? ?? ?? ??"    # mov eax, immediate value

@type  hexa: str
@param hexa: Pattern to search for.

### Function: search_process(process, pattern, minAddr, maxAddr, bufferPages, overlapping)

**Description:** Search for the given pattern within the process memory.

@type  process: L{Process}
@param process: Process to search.

@type  pattern: L{Pattern}
@param pattern: Pattern to search for.
    It must be an instance of a subclass of L{Pattern}.

    The following L{Pattern} subclasses are provided by WinAppDbg:
     - L{BytePattern}
     - L{TextPattern}
     - L{RegExpPattern}
     - L{HexPattern}

    You can also write your own subclass of L{Pattern} for customized
    searches.

@type  minAddr: int
@param minAddr: (Optional) Start the search at this memory address.

@type  maxAddr: int
@param maxAddr: (Optional) Stop the search at this memory address.

@type  bufferPages: int
@param bufferPages: (Optional) Number of memory pages to buffer when
    performing the search. Valid values are:
     - C{0} or C{None}:
       Automatically determine the required buffer size. May not give
       complete results for regular expressions that match variable
       sized strings.
     - C{> 0}: Set the buffer size, in memory pages.
     - C{< 0}: Disable buffering entirely. This may give you a little
       speed gain at the cost of an increased memory usage. If the
       target process has very large contiguous memory regions it may
       actually be slower or even fail. It's also the only way to
       guarantee complete results for regular expressions that match
       variable sized strings.

@type  overlapping: bool
@param overlapping: C{True} to allow overlapping results, C{False}
    otherwise.

    Overlapping results yield the maximum possible number of results.

    For example, if searching for "AAAA" within "AAAAAAAA" at address
    C{0x10000}, when overlapping is turned off the following matches
    are yielded::
        (0x10000, 4, "AAAA")
        (0x10004, 4, "AAAA")

    If overlapping is turned on, the following matches are yielded::
        (0x10000, 4, "AAAA")
        (0x10001, 4, "AAAA")
        (0x10002, 4, "AAAA")
        (0x10003, 4, "AAAA")
        (0x10004, 4, "AAAA")

    As you can see, the middle results are overlapping the last two.

@rtype:  iterator of tuple( int, int, str )
@return: An iterator of tuples. Each tuple contains the following:
     - The memory address where the pattern was found.
     - The size of the data that matches the pattern.
     - The data that matches the pattern.

@raise WindowsError: An error occurred when querying or reading the
    process memory.

### Function: extract_ascii_strings(cls, process, minSize, maxSize)

**Description:** Extract ASCII strings from the process memory.

@type  process: L{Process}
@param process: Process to search.

@type  minSize: int
@param minSize: (Optional) Minimum size of the strings to search for.

@type  maxSize: int
@param maxSize: (Optional) Maximum size of the strings to search for.

@rtype:  iterator of tuple(int, int, str)
@return: Iterator of strings extracted from the process memory.
    Each tuple contains the following:
     - The memory address where the string was found.
     - The size of the string.
     - The string.
