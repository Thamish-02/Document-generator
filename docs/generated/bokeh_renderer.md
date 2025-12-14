## AI Summary

A file named bokeh_renderer.py.


## Class: BokehRenderer

**Description:** Utility renderer using Bokeh to render a grid of plots over the same (x, y) range.

Args:
    nrows (int, optional): Number of rows of plots, default ``1``.
    ncols (int, optional): Number of columns of plots, default ``1``.
    figsize (tuple(float, float), optional): Figure size in inches (assuming 100 dpi), default
        ``(9, 9)``.
    show_frame (bool, optional): Whether to show frame and axes ticks, default ``True``.
    want_svg (bool, optional): Whether output is required in SVG format or not, default
        ``False``.

Warning:
    :class:`~.BokehRenderer`, unlike :class:`~.MplRenderer`, needs to be told in advance if
    output to SVG format will be required later, otherwise it will assume PNG output.

### Function: __init__(self, nrows, ncols, figsize, show_frame, want_svg)

### Function: _convert_color(self, color)

### Function: _get_figure(self, ax)

### Function: filled(self, filled, fill_type, ax, color, alpha)

**Description:** Plot filled contours on a single plot.

Args:
    filled (sequence of arrays): Filled contour data as returned by
        :meth:`~.ContourGenerator.filled`.
    fill_type (FillType or str): Type of :meth:`~.ContourGenerator.filled` data as returned
        by :attr:`~.ContourGenerator.fill_type`, or a string equivalent.
    ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
    color (str, optional): Color to plot with. May be a string color or the letter ``"C"``
        followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
        ``Category10`` palette. Default ``"C0"``.
    alpha (float, optional): Opacity to plot with, default ``0.7``.

### Function: grid(self, x, y, ax, color, alpha, point_color, quad_as_tri_alpha)

**Description:** Plot quad grid lines on a single plot.

Args:
    x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
    y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
    ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
    color (str, optional): Color to plot grid lines, default ``"black"``.
    alpha (float, optional): Opacity to plot lines with, default ``0.1``.
    point_color (str, optional): Color to plot grid points or ``None`` if grid points
        should not be plotted, default ``None``.
    quad_as_tri_alpha (float, optional): Opacity to plot ``quad_as_tri`` grid, default
        ``0``.

Colors may be a string color or the letter ``"C"`` followed by an integer in the range
``"C0"`` to ``"C9"`` to use a color from the ``Category10`` palette.

Warning:
    ``quad_as_tri_alpha > 0`` plots all quads as though they are unmasked.

### Function: lines(self, lines, line_type, ax, color, alpha, linewidth)

**Description:** Plot contour lines on a single plot.

Args:
    lines (sequence of arrays): Contour line data as returned by
        :meth:`~.ContourGenerator.lines`.
    line_type (LineType or str): Type of :meth:`~.ContourGenerator.lines` data as returned
        by :attr:`~.ContourGenerator.line_type`, or a string equivalent.
    ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
    color (str, optional): Color to plot lines. May be a string color or the letter ``"C"``
        followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
        ``Category10`` palette. Default ``"C0"``.
    alpha (float, optional): Opacity to plot lines with, default ``1.0``.
    linewidth (float, optional): Width of lines, default ``1``.

Note:
    Assumes all lines are open line strips not closed line loops.

### Function: mask(self, x, y, z, ax, color)

**Description:** Plot masked out grid points as circles on a single plot.

Args:
    x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
    y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
    z (masked array of shape (ny, nx): z-values.
    ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
    color (str, optional): Circle color, default ``"black"``.

### Function: save(self, filename, transparent)

**Description:** Save plots to SVG or PNG file.

Args:
    filename (str): Filename to save to.
    transparent (bool, optional): Whether background should be transparent, default
        ``False``.
    webdriver (WebDriver, optional): Selenium WebDriver instance to use to create the image.

        .. versionadded:: 1.1.1

Warning:
    To output to SVG file, ``want_svg=True`` must have been passed to the constructor.

### Function: save_to_buffer(self)

**Description:** Save plots to an ``io.BytesIO`` buffer.

Args:
    webdriver (WebDriver, optional): Selenium WebDriver instance to use to create the image.

        .. versionadded:: 1.1.1

Return:
    BytesIO: PNG image buffer.

### Function: show(self)

**Description:** Show plots in web browser, in usual Bokeh manner.
        

### Function: title(self, title, ax, color)

**Description:** Set the title of a single plot.

Args:
    title (str): Title text.
    ax (int or Bokeh Figure, optional): Which plot to set the title of, default ``0``.
    color (str, optional): Color to set title. May be a string color or the letter ``"C"``
        followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
        ``Category10`` palette. Default ``None`` which is ``black``.

### Function: z_values(self, x, y, z, ax, color, fmt, quad_as_tri)

**Description:** Show ``z`` values on a single plot.

Args:
    x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
    y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
    z (array-like of shape (ny, nx): z-values.
    ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
    color (str, optional): Color of added text. May be a string color or the letter ``"C"``
        followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
        ``Category10`` palette. Default ``"green"``.
    fmt (str, optional): Format to display z-values, default ``".1f"``.
    quad_as_tri (bool, optional): Whether to show z-values at the ``quad_as_tri`` centres
        of quads.

Warning:
    ``quad_as_tri=True`` shows z-values for all quads, even if masked.
