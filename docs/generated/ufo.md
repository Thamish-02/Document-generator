## AI Summary

A file named ufo.py.


### Function: zip()

**Description:** Ensure each argument to zip has the same length. Also make sure a list is
returned for python 2/3 compatibility.

## Class: GetSegmentsPen

**Description:** Pen to collect segments into lists of points for conversion.

Curves always include their initial on-curve point, so some points are
duplicated between segments.

### Function: _get_segments(glyph)

**Description:** Get a glyph's segments as extracted by GetSegmentsPen.

### Function: _set_segments(glyph, segments, reverse_direction)

**Description:** Draw segments as extracted by GetSegmentsPen back to a glyph.

### Function: _segments_to_quadratic(segments, max_err, stats, all_quadratic)

**Description:** Return quadratic approximations of cubic segments.

### Function: _glyphs_to_quadratic(glyphs, max_err, reverse_direction, stats, all_quadratic)

**Description:** Do the actual conversion of a set of compatible glyphs, after arguments
have been set up.

Return True if the glyphs were modified, else return False.

### Function: glyphs_to_quadratic(glyphs, max_err, reverse_direction, stats, all_quadratic)

**Description:** Convert the curves of a set of compatible of glyphs to quadratic.

All curves will be converted to quadratic at once, ensuring interpolation
compatibility. If this is not required, calling glyphs_to_quadratic with one
glyph at a time may yield slightly more optimized results.

Return True if glyphs were modified, else return False.

Raises IncompatibleGlyphsError if glyphs have non-interpolatable outlines.

### Function: fonts_to_quadratic(fonts, max_err_em, max_err, reverse_direction, stats, dump_stats, remember_curve_type, all_quadratic)

**Description:** Convert the curves of a collection of fonts to quadratic.

All curves will be converted to quadratic at once, ensuring interpolation
compatibility. If this is not required, calling fonts_to_quadratic with one
font at a time may yield slightly more optimized results.

Return the set of modified glyph names if any, else return an empty set.

By default, cu2qu stores the curve type in the fonts' lib, under a private
key "com.github.googlei18n.cu2qu.curve_type", and will not try to convert
them again if the curve type is already set to "quadratic".
Setting 'remember_curve_type' to False disables this optimization.

Raises IncompatibleFontsError if same-named glyphs from different fonts
have non-interpolatable outlines.

### Function: glyph_to_quadratic(glyph)

**Description:** Convenience wrapper around glyphs_to_quadratic, for just one glyph.
Return True if the glyph was modified, else return False.

### Function: font_to_quadratic(font)

**Description:** Convenience wrapper around fonts_to_quadratic, for just one font.
Return the set of modified glyph names if any, else return empty set.

### Function: __init__(self)

### Function: _add_segment(self, tag)

### Function: moveTo(self, pt)

### Function: lineTo(self, pt)

### Function: qCurveTo(self)

### Function: curveTo(self)

### Function: closePath(self)

### Function: endPath(self)

### Function: addComponent(self, glyphName, transformation)
