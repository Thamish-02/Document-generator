## AI Summary

A file named _constrained_layout.py.


### Function: do_constrained_layout(fig, h_pad, w_pad, hspace, wspace, rect, compress)

**Description:** Do the constrained_layout.  Called at draw time in
 ``figure.constrained_layout()``

Parameters
----------
fig : `~matplotlib.figure.Figure`
    `.Figure` instance to do the layout in.

h_pad, w_pad : float
  Padding around the Axes elements in figure-normalized units.

hspace, wspace : float
   Fraction of the figure to dedicate to space between the
   Axes.  These are evenly spread between the gaps between the Axes.
   A value of 0.2 for a three-column layout would have a space
   of 0.1 of the figure width between each column.
   If h/wspace < h/w_pad, then the pads are used instead.

rect : tuple of 4 floats
    Rectangle in figure coordinates to perform constrained layout in
    [left, bottom, width, height], each from 0-1.

compress : bool
    Whether to shift Axes so that white space in between them is
    removed. This is useful for simple grids of fixed-aspect Axes (e.g.
    a grid of images).

Returns
-------
layoutgrid : private debugging structure

### Function: make_layoutgrids(fig, layoutgrids, rect)

**Description:** Make the layoutgrid tree.

(Sub)Figures get a layoutgrid so we can have figure margins.

Gridspecs that are attached to Axes get a layoutgrid so Axes
can have margins.

### Function: make_layoutgrids_gs(layoutgrids, gs)

**Description:** Make the layoutgrid for a gridspec (and anything nested in the gridspec)

### Function: check_no_collapsed_axes(layoutgrids, fig)

**Description:** Check that no Axes have collapsed to zero size.

### Function: compress_fixed_aspect(layoutgrids, fig)

### Function: get_margin_from_padding(obj)

### Function: make_layout_margins(layoutgrids, fig, renderer)

**Description:** For each Axes, make a margin between the *pos* layoutbox and the
*axes* layoutbox be a minimum size that can accommodate the
decorations on the axis.

Then make room for colorbars.

Parameters
----------
layoutgrids : dict
fig : `~matplotlib.figure.Figure`
    `.Figure` instance to do the layout in.
renderer : `~matplotlib.backend_bases.RendererBase` subclass.
    The renderer to use.
w_pad, h_pad : float, default: 0
    Width and height padding (in fraction of figure).
hspace, wspace : float, default: 0
    Width and height padding as fraction of figure size divided by
    number of columns or rows.

### Function: make_margin_suptitles(layoutgrids, fig, renderer)

### Function: match_submerged_margins(layoutgrids, fig)

**Description:**     Make the margins that are submerged inside an Axes the same size.

    This allows Axes that span two columns (or rows) that are offset
    from one another to have the same size.

    This gives the proper layout for something like::
        fig = plt.figure(constrained_layout=True)
        axs = fig.subplot_mosaic("AAAB
CCDD")

    Without this routine, the Axes D will be wider than C, because the
    margin width between the two columns in C has no width by default,
    whereas the margins between the two columns of D are set by the
    width of the margin between A and B. However, obviously the user would
    like C and D to be the same size, so we need to add constraints to these
    "submerged" margins.

    This routine makes all the interior margins the same, and the spacing
    between the three columns in A and the two column in C are all set to the
    margins between the two columns of D.

    See test_constrained_layout::test_constrained_layout12 for an example.
    

### Function: get_cb_parent_spans(cbax)

**Description:** Figure out which subplotspecs this colorbar belongs to.

Parameters
----------
cbax : `~matplotlib.axes.Axes`
    Axes for the colorbar.

### Function: get_pos_and_bbox(ax, renderer)

**Description:** Get the position and the bbox for the Axes.

Parameters
----------
ax : `~matplotlib.axes.Axes`
renderer : `~matplotlib.backend_bases.RendererBase` subclass.

Returns
-------
pos : `~matplotlib.transforms.Bbox`
    Position in figure coordinates.
bbox : `~matplotlib.transforms.Bbox`
    Tight bounding box in figure coordinates.

### Function: reposition_axes(layoutgrids, fig, renderer)

**Description:** Reposition all the Axes based on the new inner bounding box.

### Function: reposition_colorbar(layoutgrids, cbax, renderer)

**Description:** Place the colorbar in its new place.

Parameters
----------
layoutgrids : dict
cbax : `~matplotlib.axes.Axes`
    Axes for the colorbar.
renderer : `~matplotlib.backend_bases.RendererBase` subclass.
    The renderer to use.
offset : array-like
    Offset the colorbar needs to be pushed to in order to
    account for multiple colorbars.

### Function: reset_margins(layoutgrids, fig)

**Description:** Reset the margins in the layoutboxes of *fig*.

Margins are usually set as a minimum, so if the figure gets smaller
the minimum needs to be zero in order for it to grow again.

### Function: colorbar_get_pad(layoutgrids, cax)
