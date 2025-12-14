## AI Summary

A file named removeOverlaps.py.


## Class: RemoveOverlapsError

### Function: skPathFromGlyph(glyphName, glyphSet)

### Function: skPathFromGlyphComponent(component, glyphSet)

### Function: componentsOverlap(glyph, glyphSet)

### Function: ttfGlyphFromSkPath(path)

### Function: _charString_from_SkPath(path, charString)

### Function: _round_path(path, round)

### Function: _simplify(path, debugGlyphName)

### Function: _same_path(path1, path2)

### Function: removeTTGlyphOverlaps(glyphName, glyphSet, glyfTable, hmtxTable, removeHinting)

### Function: _remove_glyf_overlaps()

### Function: _remove_charstring_overlaps()

### Function: _remove_cff_overlaps()

### Function: removeOverlaps(font, glyphNames, removeHinting, ignoreErrors)

**Description:** Simplify glyphs in TTFont by merging overlapping contours.

Overlapping components are first decomposed to simple contours, then merged.

Currently this only works for fonts with 'glyf' or 'CFF ' tables.
Raises NotImplementedError if 'glyf' or 'CFF ' tables are absent.

Note that removing overlaps invalidates the hinting. By default we drop hinting
from all glyphs whether or not overlaps are removed from a given one, as it would
look weird if only some glyphs are left (un)hinted.

Args:
    font: input TTFont object, modified in place.
    glyphNames: optional iterable of glyph names (str) to remove overlaps from.
        By default, all glyphs in the font are processed.
    removeHinting (bool): set to False to keep hinting for unmodified glyphs.
    ignoreErrors (bool): set to True to ignore errors while removing overlaps,
        thus keeping the tricky glyphs unchanged (fonttools/fonttools#2363).
    removeUnusedSubroutines (bool): set to False to keep unused subroutines
        in CFF table after removing overlaps. Default is to remove them if
        any glyphs are modified.

### Function: main(args)

**Description:** Simplify glyphs in TTFont by merging overlapping contours.

### Function: _get_nth_component_path(index)
