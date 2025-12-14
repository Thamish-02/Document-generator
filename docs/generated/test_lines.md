## AI Summary

A file named test_lines.py.


### Function: test_segment_hits()

**Description:** Test a problematic case.

### Function: test_invisible_Line_rendering()

**Description:** GitHub issue #1256 identified a bug in Line.draw method

Despite visibility attribute set to False, the draw method was not
returning early enough and some pre-rendering code was executed
though not necessary.

Consequence was an excessive draw time for invisible Line instances
holding a large number of points (Npts> 10**6)

### Function: test_set_line_coll_dash()

### Function: test_invalid_line_data()

### Function: test_line_dashes()

### Function: test_line_colors()

### Function: test_valid_colors()

### Function: test_linestyle_variants()

### Function: test_valid_linestyles()

### Function: test_drawstyle_variants()

### Function: test_no_subslice_with_transform(fig_ref, fig_test)

### Function: test_valid_drawstyles()

### Function: test_set_drawstyle()

### Function: test_set_line_coll_dash_image()

### Function: test_marker_fill_styles()

### Function: test_markerfacecolor_fillstyle()

**Description:** Test that markerfacecolor does not override fillstyle='none'.

### Function: test_lw_scaling()

### Function: test_is_sorted_and_has_non_nan()

### Function: test_step_markers(fig_test, fig_ref)

### Function: test_markevery(fig_test, fig_ref, parent)

### Function: test_markevery_figure_line_unsupported_relsize()

### Function: test_marker_as_markerstyle()

### Function: test_striped_lines()

### Function: test_odd_dashes(fig_test, fig_ref)

### Function: test_picking()

### Function: test_input_copy(fig_test, fig_ref)

### Function: test_markevery_prop_cycle(fig_test, fig_ref)

**Description:** Test that we can set markevery prop_cycle.

### Function: test_axline_setters()

### Function: test_axline_small_slope()

**Description:** Test that small slopes are not coerced to zero in the transform.

### Function: add_test(x, y)

### Function: add_ref(x, y)

### Function: add_test(x, y)

### Function: add_ref(x, y)
