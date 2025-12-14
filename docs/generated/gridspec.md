## AI Summary

A file named gridspec.py.


## Class: GridSpecBase

**Description:** A base class of GridSpec that specifies the geometry of the grid
that a subplot will be placed.

## Class: GridSpec

**Description:** A grid layout to place subplots within a figure.

The location of the grid cells is determined in a similar way to
`.SubplotParams` using *left*, *right*, *top*, *bottom*, *wspace*
and *hspace*.

Indexing a GridSpec instance returns a `.SubplotSpec`.

## Class: GridSpecFromSubplotSpec

**Description:** GridSpec whose subplot layout parameters are inherited from the
location specified by a given SubplotSpec.

## Class: SubplotSpec

**Description:** The location of a subplot in a `GridSpec`.

.. note::

    Likely, you will never instantiate a `SubplotSpec` yourself. Instead,
    you will typically obtain one from a `GridSpec` using item-access.

Parameters
----------
gridspec : `~matplotlib.gridspec.GridSpec`
    The GridSpec, which the subplot is referencing.
num1, num2 : int
    The subplot will occupy the *num1*-th cell of the given
    *gridspec*.  If *num2* is provided, the subplot will span between
    *num1*-th cell and *num2*-th cell **inclusive**.

    The index starts from 0.

## Class: SubplotParams

**Description:** Parameters defining the positioning of a subplots grid in a figure.

### Function: __init__(self, nrows, ncols, height_ratios, width_ratios)

**Description:** Parameters
----------
nrows, ncols : int
    The number of rows and columns of the grid.
width_ratios : array-like of length *ncols*, optional
    Defines the relative widths of the columns. Each column gets a
    relative width of ``width_ratios[i] / sum(width_ratios)``.
    If not given, all columns will have the same width.
height_ratios : array-like of length *nrows*, optional
    Defines the relative heights of the rows. Each row gets a
    relative height of ``height_ratios[i] / sum(height_ratios)``.
    If not given, all rows will have the same height.

### Function: __repr__(self)

### Function: get_geometry(self)

**Description:** Return a tuple containing the number of rows and columns in the grid.

### Function: get_subplot_params(self, figure)

### Function: new_subplotspec(self, loc, rowspan, colspan)

**Description:** Create and return a `.SubplotSpec` instance.

Parameters
----------
loc : (int, int)
    The position of the subplot in the grid as
    ``(row_index, column_index)``.
rowspan, colspan : int, default: 1
    The number of rows and columns the subplot should span in the grid.

### Function: set_width_ratios(self, width_ratios)

**Description:** Set the relative widths of the columns.

*width_ratios* must be of length *ncols*. Each column gets a relative
width of ``width_ratios[i] / sum(width_ratios)``.

### Function: get_width_ratios(self)

**Description:** Return the width ratios.

This is *None* if no width ratios have been set explicitly.

### Function: set_height_ratios(self, height_ratios)

**Description:** Set the relative heights of the rows.

*height_ratios* must be of length *nrows*. Each row gets a relative
height of ``height_ratios[i] / sum(height_ratios)``.

### Function: get_height_ratios(self)

**Description:** Return the height ratios.

This is *None* if no height ratios have been set explicitly.

### Function: get_grid_positions(self, fig)

**Description:** Return the positions of the grid cells in figure coordinates.

Parameters
----------
fig : `~matplotlib.figure.Figure`
    The figure the grid should be applied to. The subplot parameters
    (margins and spacing between subplots) are taken from *fig*.

Returns
-------
bottoms, tops, lefts, rights : array
    The bottom, top, left, right positions of the grid cells in
    figure coordinates.

### Function: _check_gridspec_exists(figure, nrows, ncols)

**Description:** Check if the figure already has a gridspec with these dimensions,
or create a new one

### Function: __getitem__(self, key)

**Description:** Create and return a `.SubplotSpec` instance.

### Function: subplots(self)

**Description:** Add all subplots specified by this `GridSpec` to its parent figure.

See `.Figure.subplots` for detailed documentation.

### Function: __init__(self, nrows, ncols, figure, left, bottom, right, top, wspace, hspace, width_ratios, height_ratios)

**Description:** Parameters
----------
nrows, ncols : int
    The number of rows and columns of the grid.

figure : `.Figure`, optional
    Only used for constrained layout to create a proper layoutgrid.

left, right, top, bottom : float, optional
    Extent of the subplots as a fraction of figure width or height.
    Left cannot be larger than right, and bottom cannot be larger than
    top. If not given, the values will be inferred from a figure or
    rcParams at draw time. See also `GridSpec.get_subplot_params`.

wspace : float, optional
    The amount of width reserved for space between subplots,
    expressed as a fraction of the average axis width.
    If not given, the values will be inferred from a figure or
    rcParams when necessary. See also `GridSpec.get_subplot_params`.

hspace : float, optional
    The amount of height reserved for space between subplots,
    expressed as a fraction of the average axis height.
    If not given, the values will be inferred from a figure or
    rcParams when necessary. See also `GridSpec.get_subplot_params`.

width_ratios : array-like of length *ncols*, optional
    Defines the relative widths of the columns. Each column gets a
    relative width of ``width_ratios[i] / sum(width_ratios)``.
    If not given, all columns will have the same width.

height_ratios : array-like of length *nrows*, optional
    Defines the relative heights of the rows. Each row gets a
    relative height of ``height_ratios[i] / sum(height_ratios)``.
    If not given, all rows will have the same height.

### Function: update(self)

**Description:** Update the subplot parameters of the grid.

Parameters that are not explicitly given are not changed. Setting a
parameter to *None* resets it to :rc:`figure.subplot.*`.

Parameters
----------
left, right, top, bottom : float or None, optional
    Extent of the subplots as a fraction of figure width or height.
wspace, hspace : float, optional
    Spacing between the subplots as a fraction of the average subplot
    width / height.

### Function: get_subplot_params(self, figure)

**Description:** Return the `.SubplotParams` for the GridSpec.

In order of precedence the values are taken from

- non-*None* attributes of the GridSpec
- the provided *figure*
- :rc:`figure.subplot.*`

Note that the ``figure`` attribute of the GridSpec is always ignored.

### Function: locally_modified_subplot_params(self)

**Description:** Return a list of the names of the subplot parameters explicitly set
in the GridSpec.

This is a subset of the attributes of `.SubplotParams`.

### Function: tight_layout(self, figure, renderer, pad, h_pad, w_pad, rect)

**Description:** Adjust subplot parameters to give specified padding.

Parameters
----------
figure : `.Figure`
    The figure.
renderer :  `.RendererBase` subclass, optional
    The renderer to be used.
pad : float
    Padding between the figure edge and the edges of subplots, as a
    fraction of the font-size.
h_pad, w_pad : float, optional
    Padding (height/width) between edges of adjacent subplots.
    Defaults to *pad*.
rect : tuple (left, bottom, right, top), default: None
    (left, bottom, right, top) rectangle in normalized figure
    coordinates that the whole subplots area (including labels) will
    fit into. Default (None) is the whole figure.

### Function: __init__(self, nrows, ncols, subplot_spec, wspace, hspace, height_ratios, width_ratios)

**Description:** Parameters
----------
nrows, ncols : int
    Number of rows and number of columns of the grid.
subplot_spec : SubplotSpec
    Spec from which the layout parameters are inherited.
wspace, hspace : float, optional
    See `GridSpec` for more details. If not specified default values
    (from the figure or rcParams) are used.
height_ratios : array-like of length *nrows*, optional
    See `GridSpecBase` for details.
width_ratios : array-like of length *ncols*, optional
    See `GridSpecBase` for details.

### Function: get_subplot_params(self, figure)

**Description:** Return a dictionary of subplot layout parameters.

### Function: get_topmost_subplotspec(self)

**Description:** Return the topmost `.SubplotSpec` instance associated with the subplot.

### Function: __init__(self, gridspec, num1, num2)

### Function: __repr__(self)

### Function: _from_subplot_args(figure, args)

**Description:** Construct a `.SubplotSpec` from a parent `.Figure` and either

- a `.SubplotSpec` -- returned as is;
- one or three numbers -- a MATLAB-style subplot specifier.

### Function: num2(self)

### Function: num2(self, value)

### Function: get_gridspec(self)

### Function: get_geometry(self)

**Description:** Return the subplot geometry as tuple ``(n_rows, n_cols, start, stop)``.

The indices *start* and *stop* define the range of the subplot within
the `GridSpec`. *stop* is inclusive (i.e. for a single cell
``start == stop``).

### Function: rowspan(self)

**Description:** The rows spanned by this subplot, as a `range` object.

### Function: colspan(self)

**Description:** The columns spanned by this subplot, as a `range` object.

### Function: is_first_row(self)

### Function: is_last_row(self)

### Function: is_first_col(self)

### Function: is_last_col(self)

### Function: get_position(self, figure)

**Description:** Update the subplot position from ``figure.subplotpars``.

### Function: get_topmost_subplotspec(self)

**Description:** Return the topmost `SubplotSpec` instance associated with the subplot.

### Function: __eq__(self, other)

**Description:** Two SubplotSpecs are considered equal if they refer to the same
position(s) in the same `GridSpec`.

### Function: __hash__(self)

### Function: subgridspec(self, nrows, ncols)

**Description:** Create a GridSpec within this subplot.

The created `.GridSpecFromSubplotSpec` will have this `SubplotSpec` as
a parent.

Parameters
----------
nrows : int
    Number of rows in grid.

ncols : int
    Number of columns in grid.

Returns
-------
`.GridSpecFromSubplotSpec`

Other Parameters
----------------
**kwargs
    All other parameters are passed to `.GridSpecFromSubplotSpec`.

See Also
--------
matplotlib.pyplot.subplots

Examples
--------
Adding three subplots in the space occupied by a single subplot::

    fig = plt.figure()
    gs0 = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    gssub = gs0[2].subgridspec(1, 3)
    for i in range(3):
        fig.add_subplot(gssub[0, i])

### Function: __init__(self, left, bottom, right, top, wspace, hspace)

**Description:** Defaults are given by :rc:`figure.subplot.[name]`.

Parameters
----------
left : float
    The position of the left edge of the subplots,
    as a fraction of the figure width.
right : float
    The position of the right edge of the subplots,
    as a fraction of the figure width.
bottom : float
    The position of the bottom edge of the subplots,
    as a fraction of the figure height.
top : float
    The position of the top edge of the subplots,
    as a fraction of the figure height.
wspace : float
    The width of the padding between subplots,
    as a fraction of the average Axes width.
hspace : float
    The height of the padding between subplots,
    as a fraction of the average Axes height.

### Function: update(self, left, bottom, right, top, wspace, hspace)

**Description:** Update the dimensions of the passed parameters. *None* means unchanged.

### Function: _normalize(key, size, axis)
