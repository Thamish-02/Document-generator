## AI Summary

A file named backend_ps.py.


### Function: _nums_to_str()

### Function: _move_path_to_path_or_stream(src, dst)

**Description:** Move the contents of file at *src* to path-or-filelike *dst*.

If *dst* is a path, the metadata of *src* are *not* copied.

### Function: _font_to_ps_type3(font_path, chars)

**Description:** Subset *chars* from the font at *font_path* into a Type 3 font.

Parameters
----------
font_path : path-like
    Path to the font to be subsetted.
chars : str
    The characters to include in the subsetted font.

Returns
-------
str
    The string representation of a Type 3 font, which can be included
    verbatim into a PostScript file.

### Function: _font_to_ps_type42(font_path, chars, fh)

**Description:** Subset *chars* from the font at *font_path* into a Type 42 font at *fh*.

Parameters
----------
font_path : path-like
    Path to the font to be subsetted.
chars : str
    The characters to include in the subsetted font.
fh : file-like
    Where to write the font.

### Function: _serialize_type42(font, subset, fontdata)

**Description:** Output a PostScript Type-42 format representation of font

Parameters
----------
font : fontTools.ttLib.ttFont.TTFont
    The original font object
subset : fontTools.ttLib.ttFont.TTFont
    The subset font object
fontdata : bytes
    The raw font data in TTF format

Returns
-------
str
    The Type-42 formatted font

### Function: _version_and_breakpoints(loca, fontdata)

**Description:** Read the version number of the font and determine sfnts breakpoints.

When a TrueType font file is written as a Type 42 font, it has to be
broken into substrings of at most 65535 bytes. These substrings must
begin at font table boundaries or glyph boundaries in the glyf table.
This function determines all possible breakpoints and it is the caller's
responsibility to do the splitting.

Helper function for _font_to_ps_type42.

Parameters
----------
loca : fontTools.ttLib._l_o_c_a.table__l_o_c_a or None
    The loca table of the font if available
fontdata : bytes
    The raw data of the font

Returns
-------
version : tuple[int, int]
    A 2-tuple of the major version number and minor version number.
breakpoints : list[int]
    The breakpoints is a sorted list of offsets into fontdata; if loca is not
    available, just the table boundaries.

### Function: _bounds(font)

**Description:** Compute the font bounding box, as if all glyphs were written
at the same start position.

Helper function for _font_to_ps_type42.

Parameters
----------
font : fontTools.ttLib.ttFont.TTFont
    The font

Returns
-------
tuple
    (xMin, yMin, xMax, yMax) of the combined bounding box
    of all the glyphs in the font

### Function: _generate_charstrings(font)

**Description:** Transform font glyphs into CharStrings

Helper function for _font_to_ps_type42.

Parameters
----------
font : fontTools.ttLib.ttFont.TTFont
    The font

Returns
-------
str
    A definition of the CharStrings dictionary in PostScript

### Function: _generate_sfnts(fontdata, font, breakpoints)

**Description:** Transform font data into PostScript sfnts format.

Helper function for _font_to_ps_type42.

Parameters
----------
fontdata : bytes
    The raw data of the font
font : fontTools.ttLib.ttFont.TTFont
    The fontTools font object
breakpoints : list
    Sorted offsets of possible breakpoints

Returns
-------
str
    The sfnts array for the font definition, consisting
    of hex-encoded strings in PostScript format

### Function: _log_if_debug_on(meth)

**Description:** Wrap `RendererPS` method *meth* to emit a PS comment with the method name,
if the global flag `debugPS` is set.

## Class: RendererPS

**Description:** The renderer handles all the drawing primitives using a graphics
context instance that controls the colors/styles.

## Class: _Orientation

## Class: FigureCanvasPS

### Function: _convert_psfrags(tmppath, psfrags, paper_width, paper_height, orientation)

**Description:** When we want to use the LaTeX backend with postscript, we write PSFrag tags
to a temporary postscript file, each one marking a position for LaTeX to
render some text. convert_psfrags generates a LaTeX document containing the
commands to convert those tags to text. LaTeX/dvips produces the postscript
file that includes the actual text.

### Function: _try_distill(func, tmppath)

### Function: gs_distill(tmpfile, eps, ptype, bbox, rotated)

**Description:** Use ghostscript's pswrite or epswrite device to distill a file.
This yields smaller files without illegal encapsulated postscript
operators. The output is low-level, converting text to outlines.

### Function: xpdf_distill(tmpfile, eps, ptype, bbox, rotated)

**Description:** Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.
This yields smaller files without illegal encapsulated postscript
operators. This distiller is preferred, generating high-level postscript
output that treats text as text.

### Function: get_bbox_header(lbrt, rotated)

**Description:** Return a postscript header string for the given bbox lbrt=(l, b, r, t).
Optionally, return rotate command.

### Function: _get_bbox_header(lbrt)

**Description:** Return a PostScript header string for bounding box *lbrt*=(l, b, r, t).

### Function: _get_rotate_command(lbrt)

**Description:** Return a PostScript 90° rotation command for bounding box *lbrt*=(l, b, r, t).

### Function: pstoeps(tmpfile, bbox, rotated)

**Description:** Convert the postscript to encapsulated postscript.  The bbox of
the eps file will be replaced with the given *bbox* argument. If
None, original bbox will be used.

## Class: _BackendPS

### Function: wrapper(self)

### Function: __init__(self, width, height, pswriter, imagedpi)

### Function: _is_transparent(self, rgb_or_rgba)

### Function: set_color(self, r, g, b, store)

### Function: set_linewidth(self, linewidth, store)

### Function: _linejoin_cmd(linejoin)

### Function: set_linejoin(self, linejoin, store)

### Function: _linecap_cmd(linecap)

### Function: set_linecap(self, linecap, store)

### Function: set_linedash(self, offset, seq, store)

### Function: set_font(self, fontname, fontsize, store)

### Function: create_hatch(self, hatch, linewidth)

### Function: get_image_magnification(self)

**Description:** Get the factor by which to magnify images passed to draw_image.
Allows a backend to have images at a different resolution to other
artists.

### Function: _convert_path(self, path, transform, clip, simplify)

### Function: _get_clip_cmd(self, gc)

### Function: draw_image(self, gc, x, y, im, transform)

### Function: draw_path(self, gc, path, transform, rgbFace)

### Function: draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace)

### Function: draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position)

### Function: draw_tex(self, gc, x, y, s, prop, angle)

### Function: draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)

### Function: draw_mathtext(self, gc, x, y, s, prop, angle)

**Description:** Draw the math text using matplotlib.mathtext.

### Function: draw_gouraud_triangles(self, gc, points, colors, trans)

### Function: _draw_ps(self, ps, gc, rgbFace)

**Description:** Emit the PostScript snippet *ps* with all the attributes from *gc*
applied.  *ps* must consist of PostScript commands to construct a path.

The *fill* and/or *stroke* kwargs can be set to False if the *ps*
string already includes filling and/or stroking, in which case
`_draw_ps` is just supplying properties and clipping.

### Function: swap_if_landscape(self, shape)

### Function: get_default_filetype(self)

### Function: _print_ps(self, fmt, outfile)

### Function: _print_figure(self, fmt, outfile)

**Description:** Render the figure to a filesystem path or a file-like object.

Parameters are as for `.print_figure`, except that *dsc_comments* is a
string containing Document Structuring Convention comments,
generated from the *metadata* parameter to `.print_figure`.

### Function: _print_figure_tex(self, fmt, outfile)

**Description:** If :rc:`text.usetex` is True, a temporary pair of tex/eps files
are created to allow tex to manage the text layout via the PSFrags
package. These files are processed to yield the final ps or eps file.

The rest of the behavior is as for `._print_figure`.

### Function: draw(self)

### Function: print_figure_impl(fh)
