## AI Summary

A file named PyColorize.py.


## Class: Parser

**Description:** Format colored Python source.
    

### Function: __init__(self, color_table, out, parent, style)

**Description:** Create a parser with a specified color table and output channel.

Call format() to process code.

### Function: format(self, raw, out, scheme)

### Function: format2(self, raw, out)

**Description:** Parse and send the colored source.

If out and scheme are not specified, the defaults (given to
constructor) are used.

out should be a file-type object. Optionally, out can be given as the
string 'str' and the parser will automatically return the output in a
string.

### Function: _inner_call_(self, toktype, toktext, start_pos)

**Description:** like call but write to a temporary buffer

### Function: __call__(self, toktype, toktext, start_pos, end_pos, line)

**Description:** Token handler, with syntax highlighting.
