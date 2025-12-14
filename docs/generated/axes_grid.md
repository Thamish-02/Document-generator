## AI Summary

A file named axes_grid.py.


## Class: CbarAxesBase

## Class: Grid

**Description:** A grid of Axes.

In Matplotlib, the Axes location (and size) is specified in normalized
figure coordinates. This may not be ideal for images that needs to be
displayed with a given aspect ratio; for example, it is difficult to
display multiple images of a same size with some fixed padding between
them.  AxesGrid can be used in such case.

Attributes
----------
axes_all : list of Axes
    A flat list of Axes. Note that you can also access this directly
    from the grid. The following is equivalent ::

        grid[i] == grid.axes_all[i]
        len(grid) == len(grid.axes_all)

axes_column : list of list of Axes
    A 2D list of Axes where the first index is the column. This results
    in the usage pattern ``grid.axes_column[col][row]``.
axes_row : list of list of Axes
    A 2D list of Axes where the first index is the row. This results
    in the usage pattern ``grid.axes_row[row][col]``.
axes_llc : Axes
    The Axes in the lower left corner.
ngrids : int
    Number of Axes in the grid.

## Class: ImageGrid

**Description:** A grid of Axes for Image display.

This class is a specialization of `~.axes_grid1.axes_grid.Grid` for displaying a
grid of images.  In particular, it forces all axes in a column to share their x-axis
and all axes in a row to share their y-axis.  It further provides helpers to add
colorbars to some or all axes.

### Function: __init__(self)

### Function: colorbar(self, mappable)

### Function: __init__(self, fig, rect, nrows_ncols, ngrids, direction, axes_pad)

**Description:** Parameters
----------
fig : `.Figure`
    The parent figure.
rect : (float, float, float, float), (int, int, int), int, or     `~.SubplotSpec`
    The axes position, as a ``(left, bottom, width, height)`` tuple,
    as a three-digit subplot position code (e.g., ``(1, 2, 1)`` or
    ``121``), or as a `~.SubplotSpec`.
nrows_ncols : (int, int)
    Number of rows and columns in the grid.
ngrids : int or None, default: None
    If not None, only the first *ngrids* axes in the grid are created.
direction : {"row", "column"}, default: "row"
    Whether axes are created in row-major ("row by row") or
    column-major order ("column by column").  This also affects the
    order in which axes are accessed using indexing (``grid[index]``).
axes_pad : float or (float, float), default: 0.02
    Padding or (horizontal padding, vertical padding) between axes, in
    inches.
share_all : bool, default: False
    Whether all axes share their x- and y-axis.  Overrides *share_x*
    and *share_y*.
share_x : bool, default: True
    Whether all axes of a column share their x-axis.
share_y : bool, default: True
    Whether all axes of a row share their y-axis.
label_mode : {"L", "1", "all", "keep"}, default: "L"
    Determines which axes will get tick labels:

    - "L": All axes on the left column get vertical tick labels;
      all axes on the bottom row get horizontal tick labels.
    - "1": Only the bottom left axes is labelled.
    - "all": All axes are labelled.
    - "keep": Do not do anything.

axes_class : subclass of `matplotlib.axes.Axes`, default: `.mpl_axes.Axes`
    The type of Axes to create.
aspect : bool, default: False
    Whether the axes aspect ratio follows the aspect ratio of the data
    limits.

### Function: _init_locators(self)

### Function: _get_col_row(self, n)

### Function: __len__(self)

### Function: __getitem__(self, i)

### Function: get_geometry(self)

**Description:** Return the number of rows and columns of the grid as (nrows, ncols).

### Function: set_axes_pad(self, axes_pad)

**Description:** Set the padding between the axes.

Parameters
----------
axes_pad : (float, float)
    The padding (horizontal pad, vertical pad) in inches.

### Function: get_axes_pad(self)

**Description:** Return the axes padding.

Returns
-------
hpad, vpad
    Padding (horizontal pad, vertical pad) in inches.

### Function: set_aspect(self, aspect)

**Description:** Set the aspect of the SubplotDivider.

### Function: get_aspect(self)

**Description:** Return the aspect of the SubplotDivider.

### Function: set_label_mode(self, mode)

**Description:** Define which axes have tick labels.

Parameters
----------
mode : {"L", "1", "all", "keep"}
    The label mode:

    - "L": All axes on the left column get vertical tick labels;
      all axes on the bottom row get horizontal tick labels.
    - "1": Only the bottom left axes is labelled.
    - "all": All axes are labelled.
    - "keep": Do not do anything.

### Function: get_divider(self)

### Function: set_axes_locator(self, locator)

### Function: get_axes_locator(self)

### Function: __init__(self, fig, rect, nrows_ncols, ngrids, direction, axes_pad)

**Description:** Parameters
----------
fig : `.Figure`
    The parent figure.
rect : (float, float, float, float) or int
    The axes position, as a ``(left, bottom, width, height)`` tuple or
    as a three-digit subplot position code (e.g., "121").
nrows_ncols : (int, int)
    Number of rows and columns in the grid.
ngrids : int or None, default: None
    If not None, only the first *ngrids* axes in the grid are created.
direction : {"row", "column"}, default: "row"
    Whether axes are created in row-major ("row by row") or
    column-major order ("column by column").  This also affects the
    order in which axes are accessed using indexing (``grid[index]``).
axes_pad : float or (float, float), default: 0.02in
    Padding or (horizontal padding, vertical padding) between axes, in
    inches.
share_all : bool, default: False
    Whether all axes share their x- and y-axis.  Note that in any case,
    all axes in a column share their x-axis and all axes in a row share
    their y-axis.
aspect : bool, default: True
    Whether the axes aspect ratio follows the aspect ratio of the data
    limits.
label_mode : {"L", "1", "all"}, default: "L"
    Determines which axes will get tick labels:

    - "L": All axes on the left column get vertical tick labels;
      all axes on the bottom row get horizontal tick labels.
    - "1": Only the bottom left axes is labelled.
    - "all": all axes are labelled.

cbar_mode : {"each", "single", "edge", None}, default: None
    Whether to create a colorbar for "each" axes, a "single" colorbar
    for the entire grid, colorbars only for axes on the "edge"
    determined by *cbar_location*, or no colorbars.  The colorbars are
    stored in the :attr:`cbar_axes` attribute.
cbar_location : {"left", "right", "bottom", "top"}, default: "right"
cbar_pad : float, default: None
    Padding between the image axes and the colorbar axes.

    .. versionchanged:: 3.10
        ``cbar_mode="single"`` no longer adds *axes_pad* between the axes
        and the colorbar if the *cbar_location* is "left" or "bottom".

cbar_size : size specification (see `.Size.from_any`), default: "5%"
    Colorbar size.
cbar_set_cax : bool, default: True
    If True, each axes in the grid has a *cax* attribute that is bound
    to associated *cbar_axes*.
axes_class : subclass of `matplotlib.axes.Axes`, default: None

### Function: _init_locators(self)
