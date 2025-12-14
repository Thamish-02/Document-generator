## AI Summary

A file named _backend_pdf_ps.py.


### Function: _cached_get_afm_from_fname(fname)

### Function: get_glyphs_subset(fontfile, characters)

**Description:** Subset a TTF font

Reads the named fontfile and restricts the font to the characters.

Parameters
----------
fontfile : str
    Path to the font file
characters : str
    Continuous set of characters to include in subset

Returns
-------
fontTools.ttLib.ttFont.TTFont
    An open font object representing the subset, which needs to
    be closed by the caller.

### Function: font_as_file(font)

**Description:** Convert a TTFont object into a file-like object.

Parameters
----------
font : fontTools.ttLib.ttFont.TTFont
    A font object

Returns
-------
BytesIO
    A file object with the font saved into it

## Class: CharacterTracker

**Description:** Helper for font subsetting by the pdf and ps backends.

Maintains a mapping of font paths to the set of character codepoints that
are being used from that font.

## Class: RendererPDFPSBase

### Function: __init__(self)

### Function: track(self, font, s)

**Description:** Record that string *s* is being typeset using font *font*.

### Function: track_glyph(self, font, glyph)

**Description:** Record that codepoint *glyph* is being typeset using font *font*.

### Function: __init__(self, width, height)

### Function: flipy(self)

### Function: option_scale_image(self)

### Function: option_image_nocomposite(self)

### Function: get_canvas_width_height(self)

### Function: get_text_width_height_descent(self, s, prop, ismath)

### Function: _get_font_afm(self, prop)

### Function: _get_font_ttf(self, prop)
