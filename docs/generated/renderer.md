## AI Summary

A file named renderer.py.


## Class: Renderer

**Description:** Abstract base class for renderers.

### Function: _grid_as_2d(self, x, y)

### Function: filled(self, filled, fill_type, ax, color, alpha)

### Function: grid(self, x, y, ax, color, alpha, point_color, quad_as_tri_alpha)

### Function: lines(self, lines, line_type, ax, color, alpha, linewidth)

### Function: mask(self, x, y, z, ax, color)

### Function: multi_filled(self, multi_filled, fill_type, ax, color)

**Description:** Plot multiple sets of filled contours on a single axes.

Args:
    multi_filled (list of filled contour arrays): Multiple filled contour sets as returned
        by :meth:`.ContourGenerator.multi_filled`.
    fill_type (FillType or str): Type of filled data as returned by
        :attr:`~.ContourGenerator.fill_type`, or string equivalent.
    ax (int or Renderer-specific axes or figure object, optional): Which axes to plot on,
        default ``0``.
    color (str or None, optional): If a string color then this same color is used for all
        filled contours. If ``None``, the default, then the filled contour sets use colors
        from the ``tab10`` colormap in order, wrapping around to the beginning if more than
        10 sets of filled contours are rendered.
    kwargs: All other keyword argument are passed on to
        :meth:`.Renderer.filled` unchanged.

.. versionadded:: 1.3.0

### Function: multi_lines(self, multi_lines, line_type, ax, color)

**Description:** Plot multiple sets of contour lines on a single axes.

Args:
    multi_lines (list of contour line arrays): Multiple contour line sets as returned by
        :meth:`.ContourGenerator.multi_lines`.
    line_type (LineType or str): Type of line data as returned by
        :attr:`~.ContourGenerator.line_type`, or string equivalent.
    ax (int or Renderer-specific axes or figure object, optional): Which axes to plot on,
        default ``0``.
    color (str or None, optional): If a string color then this same color is used for all
        lines. If ``None``, the default, then the line sets use colors from the ``tab10``
        colormap in order, wrapping around to the beginning if more than 10 sets of lines
        are rendered.
    kwargs: All other keyword argument are passed on to
        :meth:`Renderer.lines` unchanged.

.. versionadded:: 1.3.0

### Function: save(self, filename, transparent)

### Function: save_to_buffer(self)

### Function: show(self)

### Function: title(self, title, ax, color)

### Function: z_values(self, x, y, z, ax, color, fmt, quad_as_tri)
