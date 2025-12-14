## AI Summary

A file named F__e_a_t.py.


## Class: table_F__e_a_t

**Description:** Feature table

The ``Feat`` table is used exclusively by the Graphite shaping engine
to store features and possible settings specified in GDL. Graphite features
determine what rules are applied to transform a glyph stream.

Not to be confused with ``feat``, or the OpenType Layout tables
``GSUB``/``GPOS``.

See also https://graphite.sil.org/graphite_techAbout#graphite-font-tables

## Class: Feature

### Function: __init__(self, tag)

### Function: decompile(self, data, ttFont)

### Function: compile(self, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)
