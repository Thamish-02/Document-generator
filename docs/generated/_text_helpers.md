## AI Summary

A file named _text_helpers.py.


## Class: LayoutItem

### Function: warn_on_missing_glyph(codepoint, fontnames)

### Function: layout(string, font)

**Description:** Render *string* with *font*.

For each character in *string*, yield a LayoutItem instance. When such an instance
is yielded, the font's glyph is set to the corresponding character.

Parameters
----------
string : str
    The string to be rendered.
font : FT2Font
    The font.
kern_mode : Kerning
    A FreeType kerning mode.

Yields
------
LayoutItem
