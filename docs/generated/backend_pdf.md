## AI Summary

A file named backend_pdf.py.


### Function: _fill(strings, linelen)

**Description:** Make one string from sequence of strings, with whitespace in between.

The whitespace is chosen to form lines of at most *linelen* characters,
if possible.

### Function: _create_pdf_info_dict(backend, metadata)

**Description:** Create a PDF infoDict based on user-supplied metadata.

A default ``Creator``, ``Producer``, and ``CreationDate`` are added, though
the user metadata may override it. The date may be the current time, or a
time set by the ``SOURCE_DATE_EPOCH`` environment variable.

Metadata is verified to have the correct keys and their expected types. Any
unknown keys/types will raise a warning.

Parameters
----------
backend : str
    The name of the backend to use in the Producer value.

metadata : dict[str, Union[str, datetime, Name]]
    A dictionary of metadata supplied by the user with information
    following the PDF specification, also defined in
    `~.backend_pdf.PdfPages` below.

    If any value is *None*, then the key will be removed. This can be used
    to remove any pre-defined values.

Returns
-------
dict[str, Union[str, datetime, Name]]
    A validated dictionary of metadata.

### Function: _datetime_to_pdf(d)

**Description:** Convert a datetime to a PDF string representing it.

Used for PDF and PGF.

### Function: _calculate_quad_point_coordinates(x, y, width, height, angle)

**Description:** Calculate the coordinates of rectangle when rotated by angle around x, y

### Function: _get_coordinates_of_block(x, y, width, height, angle)

**Description:** Get the coordinates of rotated rectangle and rectangle that covers the
rotated rectangle.

### Function: _get_link_annotation(gc, x, y, width, height, angle)

**Description:** Create a link annotation object for embedding URLs.

### Function: pdfRepr(obj)

**Description:** Map Python objects to PDF syntax.

### Function: _font_supports_glyph(fonttype, glyph)

**Description:** Returns True if the font is able to provide codepoint *glyph* in a PDF.

For a Type 3 font, this method returns True only for single-byte
characters. For Type 42 fonts this method return True if the character is
from the Basic Multilingual Plane.

## Class: Reference

**Description:** PDF reference object.

Use PdfFile.reserveObject() to create References.

## Class: Name

**Description:** PDF name object.

## Class: Verbatim

**Description:** Store verbatim PDF command content for later inclusion in the stream.

## Class: Op

**Description:** PDF operators (not an exhaustive list).

## Class: Stream

**Description:** PDF stream object.

This has no pdfRepr method. Instead, call begin(), then output the
contents of the stream by calling write(), and finally call end().

### Function: _get_pdf_charprocs(font_path, glyph_ids)

## Class: PdfFile

**Description:** PDF file object.

## Class: RendererPdf

## Class: GraphicsContextPdf

## Class: PdfPages

**Description:** A multi-page PDF file.

Examples
--------
>>> import matplotlib.pyplot as plt
>>> # Initialize:
>>> with PdfPages('foo.pdf') as pdf:
...     # As many times as you like, create a figure fig and save it:
...     fig = plt.figure()
...     pdf.savefig(fig)
...     # When no figure is specified the current figure is saved
...     pdf.savefig()

Notes
-----
In reality `PdfPages` is a thin wrapper around `PdfFile`, in order to avoid
confusion when using `~.pyplot.savefig` and forgetting the format argument.

## Class: FigureCanvasPdf

## Class: _BackendPdf

### Function: is_string_like(x)

### Function: is_date(x)

### Function: check_trapped(x)

### Function: __init__(self, id)

### Function: __repr__(self)

### Function: pdfRepr(self)

### Function: write(self, contents, file)

### Function: __init__(self, name)

### Function: __repr__(self)

### Function: __str__(self)

### Function: __eq__(self, other)

### Function: __lt__(self, other)

### Function: __hash__(self)

### Function: pdfRepr(self)

### Function: __init__(self, x)

### Function: pdfRepr(self)

### Function: pdfRepr(self)

### Function: paint_path(cls, fill, stroke)

**Description:** Return the PDF operator to paint a path.

Parameters
----------
fill : bool
    Fill the path with the fill color.
stroke : bool
    Stroke the outline of the path with the line color.

### Function: __init__(self, id, len, file, extra, png)

**Description:** Parameters
----------
id : int
    Object id of the stream.
len : Reference or None
    An unused Reference object for the length of the stream;
    None means to use a memory buffer so the length can be inlined.
file : PdfFile
    The underlying object to write the stream to.
extra : dict from Name to anything, or None
    Extra key-value pairs to include in the stream header.
png : dict or None
    If the data is already png encoded, the decode parameters.

### Function: _writeHeader(self)

### Function: end(self)

**Description:** Finalize stream.

### Function: write(self, data)

**Description:** Write some data on the stream.

### Function: _flush(self)

**Description:** Flush the compression object.

### Function: __init__(self, filename, metadata)

**Description:** Parameters
----------
filename : str or path-like or file-like
    Output target; if a string, a file will be opened for writing.

metadata : dict from strings to strings and dates
    Information dictionary object (see PDF reference section 10.2.1
    'Document Information Dictionary'), e.g.:
    ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

    The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
    'Creator', 'Producer', 'CreationDate', 'ModDate', and
    'Trapped'. Values have been predefined for 'Creator', 'Producer'
    and 'CreationDate'. They can be removed by setting them to `None`.

### Function: newPage(self, width, height)

### Function: newTextnote(self, text, positionRect)

### Function: _get_subsetted_psname(self, ps_name, charmap)

### Function: finalize(self)

**Description:** Write out the various deferred objects and the pdf end matter.

### Function: close(self)

**Description:** Flush all buffers and free all resources.

### Function: write(self, data)

### Function: output(self)

### Function: beginStream(self, id, len, extra, png)

### Function: endStream(self)

### Function: outputStream(self, ref, data)

### Function: _write_annotations(self)

### Function: fontName(self, fontprop)

**Description:** Select a font based on fontprop and return a name suitable for
Op.selectfont. If fontprop is a string, it will be interpreted
as the filename of the font.

### Function: dviFontName(self, dvifont)

**Description:** Given a dvi font object, return a name suitable for Op.selectfont.
This registers the font information in ``self.dviFontInfo`` if not yet
registered.

### Function: writeFonts(self)

### Function: _write_afm_font(self, filename)

### Function: _embedTeXFont(self, fontinfo)

### Function: createType1Descriptor(self, t1font, fontfile)

### Function: _get_xobject_glyph_name(self, filename, glyph_name)

### Function: embedTTF(self, filename, characters)

**Description:** Embed the TTF font from the named file into the document.

### Function: alphaState(self, alpha)

**Description:** Return name of an ExtGState that sets alpha to the given value.

### Function: _soft_mask_state(self, smask)

**Description:** Return an ExtGState that sets the soft mask to the given shading.

Parameters
----------
smask : Reference
    Reference to a shading in DeviceGray color space, whose luminosity
    is to be used as the alpha channel.

Returns
-------
Name

### Function: writeExtGSTates(self)

### Function: _write_soft_mask_groups(self)

### Function: hatchPattern(self, hatch_style)

### Function: writeHatches(self)

### Function: addGouraudTriangles(self, points, colors)

**Description:** Add a Gouraud triangle shading.

Parameters
----------
points : np.ndarray
    Triangle vertices, shape (n, 3, 2)
    where n = number of triangles, 3 = vertices, 2 = x, y.
colors : np.ndarray
    Vertex colors, shape (n, 3, 1) or (n, 3, 4)
    as with points, but last dimension is either (gray,)
    or (r, g, b, alpha).

Returns
-------
Name, Reference

### Function: writeGouraudTriangles(self)

### Function: imageObject(self, image)

**Description:** Return name of an image XObject representing the given image.

### Function: _unpack(self, im)

**Description:** Unpack image array *im* into ``(data, alpha)``, which have shape
``(height, width, 3)`` (RGB) or ``(height, width, 1)`` (grayscale or
alpha), except that alpha is None if the image is fully opaque.

### Function: _writePng(self, img)

**Description:** Write the image *img* into the pdf file using png
predictors with Flate compression.

### Function: _writeImg(self, data, id, smask)

**Description:** Write the image *data*, of shape ``(height, width, 1)`` (grayscale) or
``(height, width, 3)`` (RGB), as pdf object *id* and with the soft mask
(alpha channel) *smask*, which should be either None or a ``(height,
width, 1)`` array.

### Function: writeImages(self)

### Function: markerObject(self, path, trans, fill, stroke, lw, joinstyle, capstyle)

**Description:** Return name of a marker XObject representing the given path.

### Function: writeMarkers(self)

### Function: pathCollectionObject(self, gc, path, trans, padding, filled, stroked)

### Function: writePathCollectionTemplates(self)

### Function: pathOperations(path, transform, clip, simplify, sketch)

### Function: writePath(self, path, transform, clip, sketch)

### Function: reserveObject(self, name)

**Description:** Reserve an ID for an indirect object.

The name is used for debugging in case we forget to print out
the object with writeObject.

### Function: recordXref(self, id)

### Function: writeObject(self, object, contents)

### Function: writeXref(self)

**Description:** Write out the xref table.

### Function: writeInfoDict(self)

**Description:** Write out the info dictionary, checking it for good form

### Function: writeTrailer(self)

**Description:** Write out the PDF trailer.

### Function: __init__(self, file, image_dpi, height, width)

### Function: finalize(self)

### Function: check_gc(self, gc, fillcolor)

### Function: get_image_magnification(self)

### Function: draw_image(self, gc, x, y, im, transform)

### Function: draw_path(self, gc, path, transform, rgbFace)

### Function: draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position)

### Function: draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace)

### Function: draw_gouraud_triangles(self, gc, points, colors, trans)

### Function: _setup_textpos(self, x, y, angle, oldx, oldy, oldangle)

### Function: draw_mathtext(self, gc, x, y, s, prop, angle)

### Function: draw_tex(self, gc, x, y, s, prop, angle)

### Function: encode_string(self, s, fonttype)

### Function: draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)

### Function: _draw_xobject_glyph(self, font, fontsize, glyph_idx, x, y)

**Description:** Draw a multibyte character from a Type 3 font as an XObject.

### Function: new_gc(self)

### Function: __init__(self, file)

### Function: __repr__(self)

### Function: stroke(self)

**Description:** Predicate: does the path need to be stroked (its outline drawn)?
This tests for the various conditions that disable stroking
the path, in which case it would presumably be filled.

### Function: fill(self)

**Description:** Predicate: does the path need to be filled?

An optional argument can be used to specify an alternative
_fillcolor, as needed by RendererPdf.draw_markers.

### Function: paint(self)

**Description:** Return the appropriate pdf operator to cause the path to be
stroked, filled, or both.

### Function: capstyle_cmd(self, style)

### Function: joinstyle_cmd(self, style)

### Function: linewidth_cmd(self, width)

### Function: dash_cmd(self, dashes)

### Function: alpha_cmd(self, alpha, forced, effective_alphas)

### Function: hatch_cmd(self, hatch, hatch_color, hatch_linewidth)

### Function: rgb_cmd(self, rgb)

### Function: fillcolor_cmd(self, rgb)

### Function: push(self)

### Function: pop(self)

### Function: clip_cmd(self, cliprect, clippath)

**Description:** Set clip rectangle. Calls `.pop()` and `.push()`.

### Function: delta(self, other)

**Description:** Copy properties of other into self and return PDF commands
needed to transform *self* into *other*.

### Function: copy_properties(self, other)

**Description:** Copy properties of other into self.

### Function: finalize(self)

**Description:** Make sure every pushed graphics state is popped.

### Function: __init__(self, filename, keep_empty, metadata)

**Description:** Create a new PdfPages object.

Parameters
----------
filename : str or path-like or file-like
    Plots using `PdfPages.savefig` will be written to a file at this location.
    The file is opened when a figure is saved for the first time (overwriting
    any older file with the same name).

metadata : dict, optional
    Information dictionary object (see PDF reference section 10.2.1
    'Document Information Dictionary'), e.g.:
    ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

    The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
    'Creator', 'Producer', 'CreationDate', 'ModDate', and
    'Trapped'. Values have been predefined for 'Creator', 'Producer'
    and 'CreationDate'. They can be removed by setting them to `None`.

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: _ensure_file(self)

### Function: close(self)

**Description:** Finalize this object, making the underlying file a complete
PDF file.

### Function: infodict(self)

**Description:** Return a modifiable information dictionary object
(see PDF reference section 10.2.1 'Document Information
Dictionary').

### Function: savefig(self, figure)

**Description:** Save a `.Figure` to this file as a new page.

Any other keyword arguments are passed to `~.Figure.savefig`.

Parameters
----------
figure : `.Figure` or int, default: the active figure
    The figure, or index of the figure, that is saved to the file.

### Function: get_pagecount(self)

**Description:** Return the current number of pages in the multipage pdf file.

### Function: attach_note(self, text, positionRect)

**Description:** Add a new text note to the page to be saved next. The optional
positionRect specifies the position of the new note on the
page. It is outside the page per default to make sure it is
invisible on printouts.

### Function: get_default_filetype(self)

### Function: print_pdf(self, filename)

### Function: draw(self)

### Function: toStr(n, base)

### Function: cvt(length, upe, nearest)

**Description:** Convert font coordinates to PDF glyph coordinates.

### Function: embedTTFType3(font, characters, descriptor)

**Description:** The Type 3-specific part of embedding a Truetype font

### Function: embedTTFType42(font, characters, descriptor)

**Description:** The Type 42-specific part of embedding a Truetype font

### Function: get_char_width(charcode)
