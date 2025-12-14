## AI Summary

A file named mutator.py.


### Function: interpolate_cff2_PrivateDict(topDict, interpolateFromDeltas)

### Function: interpolate_cff2_charstrings(topDict, interpolateFromDeltas, glyphOrder)

### Function: interpolate_cff2_metrics(varfont, topDict, glyphOrder, loc)

**Description:** Unlike TrueType glyphs, neither advance width nor bounding box
info is stored in a CFF2 charstring. The width data exists only in
the hmtx and HVAR tables. Since LSB data cannot be interpolated
reliably from the master LSB values in the hmtx table, we traverse
the charstring to determine the actual bound box.

### Function: instantiateVariableFont(varfont, location, inplace, overlap)

**Description:** Generate a static instance from a variable TTFont and a dictionary
defining the desired location along the variable font's axes.
The location values must be specified as user-space coordinates, e.g.:

.. code-block::

    {'wght': 400, 'wdth': 100}

By default, a new TTFont object is returned. If ``inplace`` is True, the
input varfont is modified and reduced to a static font.

When the overlap parameter is defined as True,
OVERLAP_SIMPLE and OVERLAP_COMPOUND bits are set to 1.  See
https://docs.microsoft.com/en-us/typography/opentype/spec/glyf

### Function: main(args)

**Description:** Instantiate a variation font
