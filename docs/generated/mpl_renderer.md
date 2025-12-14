## AI Summary

A file named mpl_renderer.py.


## Class: MplRenderer

**Description:** Utility renderer using Matplotlib to render a grid of plots over the same (x, y) range.

Args:
    nrows (int, optional): Number of rows of plots, default ``1``.
    ncols (int, optional): Number of columns of plots, default ``1``.
    figsize (tuple(float, float), optional): Figure size in inches, default ``(9, 9)``.
    show_frame (bool, optional): Whether to show frame and axes ticks, default ``True``.
    backend (str, optional): Matplotlib backend to use or ``None`` for default backend.
        Default ``None``.
    gridspec_kw (dict, optional): Gridspec keyword arguments to pass to ``plt.subplots``,
        default None.

## Class: MplTestRenderer

**Description:** Test renderer implemented using Matplotlib.

No whitespace around plots and no spines/ticks displayed.
Uses Agg backend, so can only save to file/buffer, cannot call ``show()``.

## Class: MplDebugRenderer

**Description:** Debug renderer implemented using Matplotlib.

Extends ``MplRenderer`` to add extra information to help in debugging such as markers, arrows,
text, etc.

### Function: __init__(self, nrows, ncols, figsize, show_frame, backend, gridspec_kw)

### Function: __del__(self)

### Function: _autoscale(self)

### Function: _get_ax(self, ax)

### Function: filled(self, filled, fill_type, ax, color, alpha)

**Description:** Plot filled contours on a single Axes.

Args:
    filled (sequence of arrays): Filled contour data as returned by
        :meth:`~.ContourGenerator.filled`.
    fill_type (FillType or str): Type of :meth:`~.ContourGenerator.filled` data as returned
        by :attr:`~.ContourGenerator.fill_type`, or string equivalent
    ax (int or Maplotlib Axes, optional): Which axes to plot on, default ``0``.
    color (str, optional): Color to plot with. May be a string color or the letter ``"C"``
        followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
        ``tab10`` colormap. Default ``"C0"``.
    alpha (float, optional): Opacity to plot with, default ``0.7``.

### Function: grid(self, x, y, ax, color, alpha, point_color, quad_as_tri_alpha)

**Description:** Plot quad grid lines on a single Axes.

Args:
    x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
    y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
    ax (int or Matplotlib Axes, optional): Which Axes to plot on, default ``0``.
    color (str, optional): Color to plot grid lines, default ``"black"``.
    alpha (float, optional): Opacity to plot lines with, default ``0.1``.
    point_color (str, optional): Color to plot grid points or ``None`` if grid points
        should not be plotted, default ``None``.
    quad_as_tri_alpha (float, optional): Opacity to plot ``quad_as_tri`` grid, default 0.

Colors may be a string color or the letter ``"C"`` followed by an integer in the range
``"C0"`` to ``"C9"`` to use a color from the ``tab10`` colormap.

Warning:
    ``quad_as_tri_alpha > 0`` plots all quads as though they are unmasked.

### Function: lines(self, lines, line_type, ax, color, alpha, linewidth)

**Description:** Plot contour lines on a single Axes.

Args:
    lines (sequence of arrays): Contour line data as returned by
        :meth:`~.ContourGenerator.lines`.
    line_type (LineType or str): Type of :meth:`~.ContourGenerator.lines` data as returned
        by :attr:`~.ContourGenerator.line_type`, or string equivalent.
    ax (int or Matplotlib Axes, optional): Which Axes to plot on, default ``0``.
    color (str, optional): Color to plot lines. May be a string color or the letter ``"C"``
        followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
        ``tab10`` colormap. Default ``"C0"``.
    alpha (float, optional): Opacity to plot lines with, default ``1.0``.
    linewidth (float, optional): Width of lines, default ``1``.

### Function: mask(self, x, y, z, ax, color)

**Description:** Plot masked out grid points as circles on a single Axes.

Args:
    x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
    y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
    z (masked array of shape (ny, nx): z-values.
    ax (int or Matplotlib Axes, optional): Which Axes to plot on, default ``0``.
    color (str, optional): Circle color, default ``"black"``.

### Function: save(self, filename, transparent)

**Description:** Save plots to SVG or PNG file.

Args:
    filename (str): Filename to save to.
    transparent (bool, optional): Whether background should be transparent, default
        ``False``.

### Function: save_to_buffer(self)

**Description:** Save plots to an ``io.BytesIO`` buffer.

Return:
    BytesIO: PNG image buffer.

### Function: show(self)

**Description:** Show plots in an interactive window, in the usual Matplotlib manner.
        

### Function: title(self, title, ax, color)

**Description:** Set the title of a single Axes.

Args:
    title (str): Title text.
    ax (int or Matplotlib Axes, optional): Which Axes to set the title of, default ``0``.
    color (str, optional): Color to set title. May be a string color or the letter ``"C"``
        followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
        ``tab10`` colormap. Default is ``None`` which uses Matplotlib's default title color
        that depends on the stylesheet in use.

### Function: z_values(self, x, y, z, ax, color, fmt, quad_as_tri)

**Description:** Show ``z`` values on a single Axes.

Args:
    x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
    y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
    z (array-like of shape (ny, nx): z-values.
    ax (int or Matplotlib Axes, optional): Which Axes to plot on, default ``0``.
    color (str, optional): Color of added text. May be a string color or the letter ``"C"``
        followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
        ``tab10`` colormap. Default ``"green"``.
    fmt (str, optional): Format to display z-values, default ``".1f"``.
    quad_as_tri (bool, optional): Whether to show z-values at the ``quad_as_tri`` centers
        of quads.

Warning:
    ``quad_as_tri=True`` shows z-values for all quads, even if masked.

### Function: __init__(self, nrows, ncols, figsize)

### Function: __init__(self, nrows, ncols, figsize, show_frame)

### Function: _arrow(self, ax, line_start, line_end, color, alpha, arrow_size)

### Function: filled(self, filled, fill_type, ax, color, alpha, line_color, line_alpha, point_color, start_point_color, arrow_size)

### Function: lines(self, lines, line_type, ax, color, alpha, linewidth, point_color, start_point_color, arrow_size)

### Function: point_numbers(self, x, y, z, ax, color)

### Function: quad_numbers(self, x, y, z, ax, color)

### Function: z_levels(self, x, y, z, lower_level, upper_level, ax, color)
