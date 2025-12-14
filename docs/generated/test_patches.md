## AI Summary

A file named test_patches.py.


### Function: test_Polygon_close()

### Function: test_corner_center()

### Function: test_ellipse_vertices()

### Function: test_rotate_rect()

### Function: test_rotate_rect_draw(fig_test, fig_ref)

### Function: test_dash_offset_patch_draw(fig_test, fig_ref)

### Function: test_negative_rect()

### Function: test_clip_to_bbox()

### Function: test_patch_alpha_coloring()

**Description:** Test checks that the patch and collection are rendered with the specified
alpha values in their facecolor and edgecolor.

### Function: test_patch_alpha_override()

### Function: test_patch_color_none()

### Function: test_patch_custom_linestyle()

### Function: test_patch_linestyle_accents()

### Function: test_patch_linestyle_none(fig_test, fig_ref)

### Function: test_wedge_movement()

### Function: test_wedge_range()

### Function: test_patch_str()

**Description:** Check that patches have nice and working `str` representation.

Note that the logic is that `__str__` is defined such that:
str(eval(str(p))) == str(p)

### Function: test_multi_color_hatch()

### Function: test_units_rectangle()

### Function: test_connection_patch()

### Function: test_connection_patch_fig(fig_test, fig_ref)

### Function: test_connection_patch_pixel_points(fig_test, fig_ref)

### Function: test_datetime_rectangle()

### Function: test_datetime_datetime_fails()

### Function: test_contains_point()

### Function: test_contains_points()

### Function: test_shadow(fig_test, fig_ref)

### Function: test_fancyarrow_units()

### Function: test_fancyarrow_setdata()

### Function: test_large_arc()

### Function: test_rotated_arcs()

### Function: test_fancyarrow_shape_error()

### Function: test_boxstyle_errors(fmt, match)

### Function: test_annulus()

### Function: test_annulus_setters()

### Function: test_annulus_setters2()

### Function: test_degenerate_polygon()

### Function: test_color_override_warning(kwarg)

### Function: test_empty_verts()

### Function: test_default_antialiased()

### Function: test_default_linestyle()

### Function: test_default_capstyle()

### Function: test_default_joinstyle()

### Function: test_autoscale_arc()

### Function: test_arc_in_collection(fig_test, fig_ref)

### Function: test_modifying_arc(fig_test, fig_ref)

### Function: test_arrow_set_data()

### Function: test_set_and_get_hatch_linewidth(fig_test, fig_ref)

### Function: test_empty_fancyarrow()
