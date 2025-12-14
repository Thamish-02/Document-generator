## AI Summary

A file named _l_t_a_g.py.


## Class: table__l_t_a_g

**Description:** Language Tag table

The AAT ``ltag`` table contains mappings between the numeric codes used
in the language field of the ``name`` table and IETF language tags.

See also https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6ltag.html

### Function: __init__(self, tag)

### Function: addTag(self, tag)

**Description:** Add 'tag' to the list of langauge tags if not already there.

Returns the integer index of 'tag' in the list of all tags.

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)
