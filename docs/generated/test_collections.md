## AI Summary

A file named test_collections.py.


### Function: pcfunc(request)

### Function: generate_EventCollection_plot()

**Description:** Generate the initial collection and plot it.

### Function: test__EventCollection__get_props()

### Function: test__EventCollection__set_positions()

### Function: test__EventCollection__add_positions()

### Function: test__EventCollection__append_positions()

### Function: test__EventCollection__extend_positions()

### Function: test__EventCollection__switch_orientation()

### Function: test__EventCollection__switch_orientation_2x()

**Description:** Check that calling switch_orientation twice sets the orientation back to
the default.

### Function: test__EventCollection__set_orientation()

### Function: test__EventCollection__set_linelength()

### Function: test__EventCollection__set_lineoffset()

### Function: test__EventCollection__set_prop()

### Function: test__EventCollection__set_color()

### Function: check_segments(coll, positions, linelength, lineoffset, orientation)

**Description:** Test helper checking that all values in the segment are correct, given a
particular set of inputs.

### Function: test_collection_norm_autoscale()

### Function: test_null_collection_datalim()

### Function: test_no_offsets_datalim()

### Function: test_add_collection()

### Function: test_collection_log_datalim(fig_test, fig_ref)

### Function: test_quiver_limits()

### Function: test_barb_limits()

### Function: test_EllipseCollection()

### Function: test_EllipseCollection_setter_getter()

### Function: test_polycollection_close()

### Function: test_regularpolycollection_rotate()

### Function: test_regularpolycollection_scale()

### Function: test_picking()

### Function: test_quadmesh_contains()

### Function: test_quadmesh_contains_concave()

### Function: test_quadmesh_cursor_data()

### Function: test_quadmesh_cursor_data_multiple_points()

### Function: test_linestyle_single_dashes()

### Function: test_size_in_xy()

### Function: test_pandas_indexing(pd)

### Function: test_lslw_bcast()

### Function: test_set_wrong_linestyle()

### Function: test_capstyle()

### Function: test_joinstyle()

### Function: test_cap_and_joinstyle_image()

### Function: test_scatter_post_alpha()

### Function: test_scatter_alpha_array()

### Function: test_pathcollection_legend_elements()

### Function: test_EventCollection_nosort()

### Function: test_collection_set_verts_array()

### Function: test_fill_between_poly_collection_set_data(fig_test, fig_ref, kwargs)

### Function: test_fill_between_poly_collection_raise(t_direction, f1, shape, where, msg)

### Function: test_collection_set_array()

### Function: test_blended_collection_autolim()

### Function: test_singleton_autolim()

### Function: test_autolim_with_zeros(transform, expected)

### Function: test_quadmesh_set_array_validation(pcfunc)

### Function: test_polyquadmesh_masked_vertices_array()

### Function: test_quadmesh_get_coordinates(pcfunc)

### Function: test_quadmesh_set_array()

### Function: test_quadmesh_vmin_vmax(pcfunc)

### Function: test_quadmesh_alpha_array(pcfunc)

### Function: test_alpha_validation(pcfunc)

### Function: test_legend_inverse_size_label_relationship()

**Description:** Ensure legend markers scale appropriately when label and size are
inversely related.
Here label = 5 / size

### Function: test_color_logic(pcfunc)

### Function: test_LineCollection_args()

### Function: test_array_dimensions(pcfunc)

### Function: test_get_segments()

### Function: test_set_offsets_late()

### Function: test_set_offset_transform()

### Function: test_set_offset_units()

### Function: test_check_masked_offsets()

### Function: test_masked_set_offsets(fig_ref, fig_test)

### Function: test_check_offsets_dtype()

### Function: test_striped_lines(fig_test, fig_ref, gapcolor)

### Function: test_hatch_linewidth(fig_test, fig_ref)

## Class: SquareCollection

### Function: __init__(self)

### Function: get_transform(self)

**Description:** Return transform scaling circle areas to data space.
