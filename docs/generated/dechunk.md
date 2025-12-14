## AI Summary

A file named dechunk.py.


### Function: dechunk_filled(filled, fill_type)

**Description:** Return the specified filled contours with chunked data moved into the first chunk.

Filled contours that are not chunked (``FillType.OuterCode`` and ``FillType.OuterOffset``) and
those that are but only contain a single chunk are returned unmodified. Individual polygons are
unchanged, they are not geometrically combined.

Args:
    filled (sequence of arrays): Filled contour data, such as returned by
        :meth:`.ContourGenerator.filled`.
    fill_type (FillType or str): Type of :meth:`~.ContourGenerator.filled` as enum or string
        equivalent.

Return:
    Filled contours in a single chunk.

.. versionadded:: 1.2.0

### Function: dechunk_lines(lines, line_type)

**Description:** Return the specified contour lines with chunked data moved into the first chunk.

Contour lines that are not chunked (``LineType.Separate`` and ``LineType.SeparateCode``) and
those that are but only contain a single chunk are returned unmodified. Individual lines are
unchanged, they are not geometrically combined.

Args:
    lines (sequence of arrays): Contour line data, such as returned by
        :meth:`.ContourGenerator.lines`.
    line_type (LineType or str): Type of :meth:`~.ContourGenerator.lines` as enum or string
        equivalent.

Return:
    Contour lines in a single chunk.

.. versionadded:: 1.2.0

### Function: dechunk_multi_filled(multi_filled, fill_type)

**Description:** Return multiple sets of filled contours with chunked data moved into the first chunks.

Filled contours that are not chunked (``FillType.OuterCode`` and ``FillType.OuterOffset``) and
those that are but only contain a single chunk are returned unmodified. Individual polygons are
unchanged, they are not geometrically combined.

Args:
    multi_filled (nested sequence of arrays): Filled contour data, such as returned by
        :meth:`.ContourGenerator.multi_filled`.
    fill_type (FillType or str): Type of :meth:`~.ContourGenerator.filled` as enum or string
        equivalent.

Return:
    Multiple sets of filled contours in a single chunk.

.. versionadded:: 1.3.0

### Function: dechunk_multi_lines(multi_lines, line_type)

**Description:** Return multiple sets of contour lines with all chunked data moved into the first chunks.

Contour lines that are not chunked (``LineType.Separate`` and ``LineType.SeparateCode``) and
those that are but only contain a single chunk are returned unmodified. Individual lines are
unchanged, they are not geometrically combined.

Args:
    multi_lines (nested sequence of arrays): Contour line data, such as returned by
        :meth:`.ContourGenerator.multi_lines`.
    line_type (LineType or str): Type of :meth:`~.ContourGenerator.lines` as enum or string
        equivalent.

Return:
    Multiple sets of contour lines in a single chunk.

.. versionadded:: 1.3.0
