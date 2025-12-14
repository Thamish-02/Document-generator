## AI Summary

A file named _t_r_a_k.py.


## Class: table__t_r_a_k

**Description:** The AAT ``trak`` table can store per-size adjustments to each glyph's
sidebearings to make when tracking is enabled, which applications can
use to provide more visually balanced line spacing.

See also https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6trak.html

## Class: TrackData

## Class: TrackTableEntry

### Function: compile(self, ttFont)

### Function: decompile(self, data, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __init__(self, initialdata)

### Function: compile(self, offset)

### Function: decompile(self, data, offset)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: sizes(self)

### Function: __getitem__(self, track)

### Function: __delitem__(self, track)

### Function: __setitem__(self, track, entry)

### Function: __len__(self)

### Function: __iter__(self)

### Function: keys(self)

### Function: __repr__(self)

### Function: __init__(self, values, nameIndex)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: __getitem__(self, size)

### Function: __delitem__(self, size)

### Function: __setitem__(self, size, value)

### Function: __len__(self)

### Function: __iter__(self)

### Function: keys(self)

### Function: __repr__(self)

### Function: __eq__(self, other)

### Function: __ne__(self, other)
