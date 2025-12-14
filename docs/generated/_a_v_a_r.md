## AI Summary

A file named _a_v_a_r.py.


## Class: table__a_v_a_r

**Description:** Axis Variations table

This class represents the ``avar`` table of a variable font. The object has one
substantive attribute, ``segments``, which maps axis tags to a segments dictionary::

    >>> font["avar"].segments   # doctest: +SKIP
    {'wght': {-1.0: -1.0,
      0.0: 0.0,
      0.125: 0.11444091796875,
      0.25: 0.23492431640625,
      0.5: 0.35540771484375,
      0.625: 0.5,
      0.75: 0.6566162109375,
      0.875: 0.81927490234375,
      1.0: 1.0},
     'ital': {-1.0: -1.0, 0.0: 0.0, 1.0: 1.0}}

Notice that the segments dictionary is made up of normalized values. A valid
``avar`` segment mapping must contain the entries ``-1.0: -1.0, 0.0: 0.0, 1.0: 1.0``.
fontTools does not enforce this, so it is your responsibility to ensure that
mappings are valid.

See also https://learn.microsoft.com/en-us/typography/opentype/spec/avar

### Function: __init__(self, tag)

### Function: compile(self, ttFont)

### Function: decompile(self, data, ttFont)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: renormalizeLocation(self, location, font, dropZeroes)
