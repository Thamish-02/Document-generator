## AI Summary

A file named backend_svg.py.


### Function: _escape_cdata(s)

### Function: _escape_comment(s)

### Function: _escape_attrib(s)

### Function: _quote_escape_attrib(s)

### Function: _short_float_fmt(x)

**Description:** Create a short string representation of a float, which is %f
formatting with trailing zeros and the decimal point removed.

## Class: XMLWriter

**Description:** Parameters
----------
file : writable text file-like object

### Function: _generate_transform(transform_list)

### Function: _generate_css(attrib)

### Function: _check_is_str(info, key)

### Function: _check_is_iterable_of_str(infos, key)

## Class: RendererSVG

## Class: FigureCanvasSVG

## Class: _BackendSVG

### Function: __init__(self, file)

### Function: __flush(self, indent)

### Function: start(self, tag, attrib)

**Description:** Open a new element.  Attributes can be given as keyword
arguments, or as a string/string dictionary. The method returns
an opaque identifier that can be passed to the :meth:`close`
method, to close all open elements up to and including this one.

Parameters
----------
tag
    Element tag.
attrib
    Attribute dictionary.  Alternatively, attributes can be given as
    keyword arguments.

Returns
-------
An element identifier.

### Function: comment(self, comment)

**Description:** Add a comment to the output stream.

Parameters
----------
comment : str
    Comment text.

### Function: data(self, text)

**Description:** Add character data to the output stream.

Parameters
----------
text : str
    Character data.

### Function: end(self, tag, indent)

**Description:** Close the current element (opened by the most recent call to
:meth:`start`).

Parameters
----------
tag
    Element tag.  If given, the tag must match the start tag.  If
    omitted, the current element is closed.
indent : bool, default: True

### Function: close(self, id)

**Description:** Close open elements, up to (and including) the element identified
by the given identifier.

Parameters
----------
id
    Element identifier, as returned by the :meth:`start` method.

### Function: element(self, tag, text, attrib)

**Description:** Add an entire element.  This is the same as calling :meth:`start`,
:meth:`data`, and :meth:`end` in sequence. The *text* argument can be
omitted.

### Function: flush(self)

**Description:** Flush the output stream.

### Function: __init__(self, width, height, svgwriter, basename, image_dpi)

### Function: _get_clippath_id(self, clippath)

**Description:** Returns a stable and unique identifier for the *clippath* argument
object within the current rendering context.

This allows plots that include custom clip paths to produce identical
SVG output on each render, provided that the :rc:`svg.hashsalt` config
setting and the ``SOURCE_DATE_EPOCH`` build-time environment variable
are set to fixed values.

### Function: finalize(self)

### Function: _write_metadata(self, metadata)

### Function: _write_default_style(self)

### Function: _make_id(self, type, content)

### Function: _make_flip_transform(self, transform)

### Function: _get_hatch(self, gc, rgbFace)

**Description:** Create a new hatch pattern

### Function: _write_hatches(self)

### Function: _get_style_dict(self, gc, rgbFace)

**Description:** Generate a style string from the GraphicsContext and rgbFace.

### Function: _get_style(self, gc, rgbFace)

### Function: _get_clip_attrs(self, gc)

### Function: _write_clips(self)

### Function: open_group(self, s, gid)

### Function: close_group(self, s)

### Function: option_image_nocomposite(self)

### Function: _convert_path(self, path, transform, clip, simplify, sketch)

### Function: draw_path(self, gc, path, transform, rgbFace)

### Function: draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace)

### Function: draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position)

### Function: _draw_gouraud_triangle(self, transformed_points, colors)

### Function: draw_gouraud_triangles(self, gc, triangles_array, colors_array, transform)

### Function: option_scale_image(self)

### Function: get_image_magnification(self)

### Function: draw_image(self, gc, x, y, im, transform)

### Function: _update_glyph_map_defs(self, glyph_map_new)

**Description:** Emit definitions for not-yet-defined glyphs, and record them as having
been defined.

### Function: _adjust_char_id(self, char_id)

### Function: _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath, mtext)

### Function: _draw_text_as_text(self, gc, x, y, s, prop, angle, ismath, mtext)

### Function: draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)

### Function: flipy(self)

### Function: get_canvas_width_height(self)

### Function: get_text_width_height_descent(self, s, prop, ismath)

### Function: print_svg(self, filename)

**Description:** Parameters
----------
filename : str or path-like or file-like
    Output target; if a string, a file will be opened for writing.

metadata : dict[str, Any], optional
    Metadata in the SVG file defined as key-value pairs of strings,
    datetimes, or lists of strings, e.g., ``{'Creator': 'My software',
    'Contributor': ['Me', 'My Friend'], 'Title': 'Awesome'}``.

    The standard keys and their value types are:

    * *str*: ``'Coverage'``, ``'Description'``, ``'Format'``,
      ``'Identifier'``, ``'Language'``, ``'Relation'``, ``'Source'``,
      ``'Title'``, and ``'Type'``.
    * *str* or *list of str*: ``'Contributor'``, ``'Creator'``,
      ``'Keywords'``, ``'Publisher'``, and ``'Rights'``.
    * *str*, *date*, *datetime*, or *tuple* of same: ``'Date'``. If a
      non-*str*, then it will be formatted as ISO 8601.

    Values have been predefined for ``'Creator'``, ``'Date'``,
    ``'Format'``, and ``'Type'``. They can be removed by setting them
    to `None`.

    Information is encoded as `Dublin Core Metadata`__.

    .. _DC: https://www.dublincore.org/specifications/dublin-core/

    __ DC_

### Function: print_svgz(self, filename)

### Function: get_default_filetype(self)

### Function: draw(self)

### Function: ensure_metadata(mid)

### Function: _normalize_sans(name)

### Function: _expand_family_entry(fn)

### Function: _get_all_quoted_names(prop)
