## AI Summary

A file named _layoutgrid.py.


## Class: LayoutGrid

**Description:** Analogous to a gridspec, and contained in another LayoutGrid.

### Function: seq_id()

**Description:** Generate a short sequential id for layoutbox objects.

### Function: plot_children(fig, lg, level)

**Description:** Simple plotting to show where boxes are.

### Function: __init__(self, parent, parent_pos, parent_inner, name, ncols, nrows, h_pad, w_pad, width_ratios, height_ratios)

### Function: __repr__(self)

### Function: reset_margins(self)

**Description:** Reset all the margins to zero.  Must do this after changing
figure size, for instance, because the relative size of the
axes labels etc changes.

### Function: add_constraints(self, parent)

### Function: hard_constraints(self)

**Description:** These are the redundant constraints, plus ones that make the
rest of the code easier.

### Function: add_child(self, child, i, j)

### Function: parent_constraints(self, parent)

### Function: grid_constraints(self)

### Function: edit_margin(self, todo, size, cell)

**Description:** Change the size of the margin for one cell.

Parameters
----------
todo : string (one of 'left', 'right', 'bottom', 'top')
    margin to alter.

size : float
    Size of the margin.  If it is larger than the existing minimum it
    updates the margin size. Fraction of figure size.

cell : int
    Cell column or row to edit.

### Function: edit_margin_min(self, todo, size, cell)

**Description:** Change the minimum size of the margin for one cell.

Parameters
----------
todo : string (one of 'left', 'right', 'bottom', 'top')
    margin to alter.

size : float
    Minimum size of the margin .  If it is larger than the
    existing minimum it updates the margin size. Fraction of
    figure size.

cell : int
    Cell column or row to edit.

### Function: edit_margins(self, todo, size)

**Description:** Change the size of all the margin of all the cells in the layout grid.

Parameters
----------
todo : string (one of 'left', 'right', 'bottom', 'top')
    margin to alter.

size : float
    Size to set the margins.  Fraction of figure size.

### Function: edit_all_margins_min(self, todo, size)

**Description:** Change the minimum size of all the margin of all
the cells in the layout grid.

Parameters
----------
todo : {'left', 'right', 'bottom', 'top'}
    The margin to alter.

size : float
    Minimum size of the margin.  If it is larger than the
    existing minimum it updates the margin size. Fraction of
    figure size.

### Function: edit_outer_margin_mins(self, margin, ss)

**Description:** Edit all four margin minimums in one statement.

Parameters
----------
margin : dict
    size of margins in a dict with keys 'left', 'right', 'bottom',
    'top'

ss : SubplotSpec
    defines the subplotspec these margins should be applied to

### Function: get_margins(self, todo, col)

**Description:** Return the margin at this position

### Function: get_outer_bbox(self, rows, cols)

**Description:** Return the outer bounding box of the subplot specs
given by rows and cols.  rows and cols can be spans.

### Function: get_inner_bbox(self, rows, cols)

**Description:** Return the inner bounding box of the subplot specs
given by rows and cols.  rows and cols can be spans.

### Function: get_bbox_for_cb(self, rows, cols)

**Description:** Return the bounding box that includes the
decorations but, *not* the colorbar...

### Function: get_left_margin_bbox(self, rows, cols)

**Description:** Return the left margin bounding box of the subplot specs
given by rows and cols.  rows and cols can be spans.

### Function: get_bottom_margin_bbox(self, rows, cols)

**Description:** Return the left margin bounding box of the subplot specs
given by rows and cols.  rows and cols can be spans.

### Function: get_right_margin_bbox(self, rows, cols)

**Description:** Return the left margin bounding box of the subplot specs
given by rows and cols.  rows and cols can be spans.

### Function: get_top_margin_bbox(self, rows, cols)

**Description:** Return the left margin bounding box of the subplot specs
given by rows and cols.  rows and cols can be spans.

### Function: update_variables(self)

**Description:** Update the variables for the solver attached to this layoutgrid.
