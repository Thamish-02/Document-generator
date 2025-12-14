## AI Summary

A file named test_axes3d.py.


### Function: plot_cuboid(ax, scale)

### Function: test_invisible_axes(fig_test, fig_ref)

### Function: test_grid_off()

### Function: test_invisible_ticks_axis()

### Function: test_axis_positions()

### Function: test_aspects()

### Function: test_aspects_adjust_box()

### Function: test_axes3d_repr()

### Function: test_axes3d_primary_views()

### Function: test_bar3d()

### Function: test_bar3d_colors()

### Function: test_bar3d_shaded()

### Function: test_bar3d_notshaded()

### Function: test_bar3d_lightsource()

### Function: test_contour3d()

### Function: test_contour3d_extend3d()

### Function: test_contourf3d()

### Function: test_contourf3d_fill()

### Function: test_contourf3d_extend(fig_test, fig_ref, extend, levels)

### Function: test_tricontour()

### Function: test_contour3d_1d_input()

### Function: test_lines3d()

### Function: test_plot_scalar(fig_test, fig_ref)

### Function: test_invalid_line_data()

### Function: test_mixedsubplots()

### Function: test_tight_layout_text(fig_test, fig_ref)

### Function: test_scatter3d()

### Function: test_scatter3d_color()

### Function: test_scatter3d_linewidth()

### Function: test_scatter3d_linewidth_modification(fig_ref, fig_test)

### Function: test_scatter3d_modification(fig_ref, fig_test)

### Function: test_scatter3d_sorting(fig_ref, fig_test, depthshade)

**Description:** Test that marker properties are correctly sorted.

### Function: test_marker_draw_order_data_reversed(fig_test, fig_ref, azim)

**Description:** Test that the draw order does not depend on the data point order.

For the given viewing angle at azim=-50, the yellow marker should be in
front. For azim=130, the blue marker should be in front.

### Function: test_marker_draw_order_view_rotated(fig_test, fig_ref)

**Description:** Test that the draw order changes with the direction.

If we rotate *azim* by 180 degrees and exchange the colors, the plot
plot should look the same again.

### Function: test_plot_3d_from_2d()

### Function: test_fill_between_quad()

### Function: test_fill_between_polygon()

### Function: test_surface3d()

### Function: test_surface3d_label_offset_tick_position()

### Function: test_surface3d_shaded()

### Function: test_surface3d_masked()

### Function: test_plot_scatter_masks(fig_test, fig_ref)

### Function: test_plot_surface_None_arg(fig_test, fig_ref)

### Function: test_surface3d_masked_strides()

### Function: test_text3d()

### Function: test_text3d_modification(fig_ref, fig_test)

### Function: test_trisurf3d()

### Function: test_trisurf3d_shaded()

### Function: test_wireframe3d()

### Function: test_wireframe3dzerocstride()

### Function: test_wireframe3dzerorstride()

### Function: test_wireframe3dzerostrideraises()

### Function: test_mixedsamplesraises()

### Function: test_quiver3d()

### Function: test_quiver3d_empty(fig_test, fig_ref)

### Function: test_quiver3d_masked()

### Function: test_quiver3d_colorcoded()

### Function: test_patch_modification()

### Function: test_patch_collection_modification(fig_test, fig_ref)

### Function: test_poly3dcollection_verts_validation()

### Function: test_poly3dcollection_closed()

### Function: test_poly_collection_2d_to_3d_empty()

### Function: test_poly3dcollection_alpha()

### Function: test_add_collection3d_zs_array()

### Function: test_add_collection3d_zs_scalar()

### Function: test_line3dCollection_autoscaling()

### Function: test_poly3dCollection_autoscaling()

### Function: test_axes3d_labelpad()

### Function: test_axes3d_cla()

### Function: test_axes3d_rotated()

### Function: test_plotsurface_1d_raises()

### Function: _test_proj_make_M()

### Function: test_proj_transform()

### Function: _test_proj_draw_axes(M, s)

### Function: test_proj_axes_cube()

### Function: test_proj_axes_cube_ortho()

### Function: test_world()

### Function: test_autoscale()

### Function: test_unautoscale(axis, auto)

### Function: test_culling(fig_test, fig_ref)

### Function: test_axes3d_focal_length_checks()

### Function: test_axes3d_focal_length()

### Function: test_axes3d_ortho()

### Function: test_axes3d_isometric()

### Function: test_axlim_clip(fig_test, fig_ref)

### Function: test_invalid_axes_limits(setter, side, value)

## Class: TestVoxels

### Function: test_line3d_set_get_data_3d()

### Function: test_inverted(fig_test, fig_ref)

### Function: test_inverted_cla()

### Function: test_ax3d_tickcolour()

### Function: test_ticklabel_format(fig_test, fig_ref)

### Function: test_quiver3D_smoke(fig_test, fig_ref)

### Function: test_minor_ticks()

### Function: test_errorbar3d_errorevery()

**Description:** Tests errorevery functionality for 3D errorbars.

### Function: test_errorbar3d()

**Description:** Tests limits, color styling, and legend for 3D errorbars.

### Function: test_stem3d()

### Function: test_equal_box_aspect()

### Function: test_colorbar_pos()

### Function: test_inverted_zaxis()

### Function: test_set_zlim()

### Function: test_shared_view(fig_test, fig_ref)

### Function: test_shared_axes_retick()

### Function: test_quaternion()

### Function: test_rotate(style)

**Description:** Test rotating using the left mouse button.

### Function: test_pan()

**Description:** Test mouse panning using the middle mouse button.

### Function: test_toolbar_zoom_pan(tool, button, key, expected)

### Function: test_scalarmap_update(fig_test, fig_ref)

### Function: test_subfigure_simple()

### Function: test_computed_zorder()

### Function: test_format_coord()

### Function: test_get_axis_position()

### Function: test_margins()

### Function: test_margin_getters()

### Function: test_margins_errors(err, args, kwargs, match)

### Function: test_text_3d(fig_test, fig_ref)

### Function: test_draw_single_lines_from_Nx1()

### Function: test_pathpatch_3d(fig_test, fig_ref)

### Function: test_scatter_spiral()

### Function: test_Poly3DCollection_get_path()

### Function: test_Poly3DCollection_get_facecolor()

### Function: test_Poly3DCollection_get_edgecolor()

### Function: test_view_init_vertical_axis(vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected)

**Description:** Test the actual projection, axis lines and ticks matches expected values.

Parameters
----------
vertical_axis : str
    Axis to align vertically.
proj_expected : ndarray
    Expected values from ax.get_proj().
axis_lines_expected : tuple of arrays
    Edgepoints of the axis line. Expected values retrieved according
    to ``ax.get_[xyz]axis().line.get_data()``.
tickdirs_expected : list of int
    indexes indicating which axis to create a tick line along.

### Function: test_on_move_vertical_axis(vertical_axis)

**Description:** Test vertical axis is respected when rotating the plot interactively.

### Function: test_set_box_aspect_vertical_axis(vertical_axis, aspect_expected)

### Function: test_arc_pathpatch()

### Function: test_panecolor_rcparams()

### Function: test_mutating_input_arrays_y_and_z(fig_test, fig_ref)

**Description:** Test to see if the `z` axis does not get mutated
after a call to `Axes3D.plot`

test cases came from GH#8990

### Function: test_scatter_masked_color()

**Description:** Test color parameter usage with non-finite coordinate arrays.

GH#26236

### Function: test_surface3d_zsort_inf()

### Function: test_Poly3DCollection_init_value_error()

### Function: test_ndarray_color_kwargs_value_error()

### Function: f(t)

### Function: test_simple(self)

### Function: test_edge_style(self)

### Function: test_named_colors(self)

**Description:** Test with colors set to a 3D object array of strings.

### Function: test_rgb_data(self)

**Description:** Test with colors set to a 4d float array of rgb data.

### Function: test_alpha(self)

### Function: test_xyz(self)

### Function: test_calling_conventions(self)

### Function: get_formatters(ax, names)

### Function: convert_lim(dmin, dmax)

**Description:** Convert min/max limits to center and range.

### Function: midpoints(x)
