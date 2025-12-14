## AI Summary

A file named width.py.


## Class: missingdict

### Function: cumSum(f, op, start, decreasing)

### Function: byteCost(widths, default, nominal)

### Function: optimizeWidthsBruteforce(widths)

**Description:** Bruteforce version.  Veeeeeeeeeeeeeeeeery slow.  Only works for smallests of fonts.

### Function: optimizeWidths(widths)

**Description:** Given a list of glyph widths, or dictionary mapping glyph width to number of
glyphs having that, returns a tuple of best CFF default and nominal glyph widths.

This algorithm is linear in UPEM+numGlyphs.

### Function: main(args)

**Description:** Calculate optimum defaultWidthX/nominalWidthX values

### Function: __init__(self, missing_func)

### Function: __missing__(self, v)
