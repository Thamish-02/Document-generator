## AI Summary

A file named line_numbers.py.


## Class: LineNumbers

**Description:** Class to convert between character offsets in a text string, and pairs (line, column) of 1-based
line and 0-based column numbers, as used by tokens and AST nodes.

This class expects unicode for input and stores positions in unicode. But it supports
translating to and from utf8 offsets, which are used by ast parsing.

### Function: __init__(self, text)

### Function: from_utf8_col(self, line, utf8_column)

**Description:** Given a 1-based line number and 0-based utf8 column, returns a 0-based unicode column.

### Function: line_to_offset(self, line, column)

**Description:** Converts 1-based line number and 0-based column to 0-based character offset into text.

### Function: offset_to_line(self, offset)

**Description:** Converts 0-based character offset to pair (line, col) of 1-based line and 0-based column
numbers.
