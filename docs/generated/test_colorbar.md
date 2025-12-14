## AI Summary

A file named test_colorbar.py.


### Function: _get_cmap_norms()

**Description:** Define a colormap and appropriate norms for each of the four
possible settings of the extend keyword.

Helper function for _colorbar_extension_shape and
colorbar_extension_length.

### Function: _colorbar_extension_shape(spacing)

**Description:** Produce 4 colorbars with rectangular extensions for either uniform
or proportional spacing.

Helper function for test_colorbar_extension_shape.

### Function: _colorbar_extension_length(spacing)

**Description:** Produce 12 colorbars with variable length extensions for either
uniform or proportional spacing.

Helper function for test_colorbar_extension_length.

### Function: test_colorbar_extension_shape()

**Description:** Test rectangular colorbar extensions.

### Function: test_colorbar_extension_length()

**Description:** Test variable length colorbar extensions.

### Function: test_colorbar_extension_inverted_axis(orientation, extend, expected)

**Description:** Test extension color with an inverted axis

### Function: test_colorbar_positioning(use_gridspec)

### Function: test_colorbar_single_ax_panchor_false()

### Function: test_colorbar_single_ax_panchor_east(constrained)

### Function: test_contour_colorbar()

### Function: test_gridspec_make_colorbar()

### Function: test_colorbar_single_scatter()

### Function: test_remove_from_figure(nested_gridspecs, use_gridspec)

**Description:** Test `remove` with the specified ``use_gridspec`` setting.

### Function: test_remove_from_figure_cl()

**Description:** Test `remove` with constrained_layout.

### Function: test_colorbarbase()

### Function: test_parentless_mappable()

### Function: test_colorbar_closed_patch()

### Function: test_colorbar_ticks()

### Function: test_colorbar_minorticks_on_off()

### Function: test_cbar_minorticks_for_rc_xyminortickvisible()

**Description:** issue gh-16468.

Making sure that minor ticks on the colorbar are turned on
(internally) using the cbar.minorticks_on() method when
rcParams['xtick.minor.visible'] = True (for horizontal cbar)
rcParams['ytick.minor.visible'] = True (for vertical cbar).
Using cbar.minorticks_on() ensures that the minor ticks
don't overflow into the extend regions of the colorbar.

### Function: test_colorbar_autoticks()

### Function: test_colorbar_autotickslog()

### Function: test_colorbar_get_ticks()

### Function: test_colorbar_lognorm_extension(extend)

### Function: test_colorbar_powernorm_extension()

### Function: test_colorbar_axes_kw()

### Function: test_colorbar_log_minortick_labels()

### Function: test_colorbar_renorm()

### Function: test_colorbar_format(fmt)

### Function: test_colorbar_scale_reset()

### Function: test_colorbar_get_ticks_2()

### Function: test_colorbar_inverted_ticks()

### Function: test_mappable_no_alpha()

### Function: test_mappable_2d_alpha()

### Function: test_colorbar_label()

**Description:** Test the label parameter. It should just be mapped to the xlabel/ylabel of
the axes, depending on the orientation.

### Function: test_keeping_xlabel()

### Function: test_colorbar_int(clim)

### Function: test_anchored_cbar_position_using_specgrid()

### Function: test_colorbar_change_lim_scale()

### Function: test_axes_handles_same_functions(fig_ref, fig_test)

### Function: test_inset_colorbar_layout()

### Function: test_twoslope_colorbar()

### Function: test_remove_cb_whose_mappable_has_no_figure(fig_ref, fig_test)

### Function: test_aspects()

### Function: test_proportional_colorbars()

### Function: test_colorbar_extend_drawedges()

### Function: test_colorbar_contourf_extend_patches()

### Function: test_negative_boundarynorm()

### Function: test_centerednorm()

### Function: test_nonorm()

### Function: test_boundaries()

### Function: test_colorbar_no_warning_rcparams_grid_true()

### Function: test_colorbar_set_formatter_locator()

### Function: test_colorbar_extend_alpha()

### Function: test_offset_text_loc()

### Function: test_title_text_loc()

### Function: test_passing_location(fig_ref, fig_test)

### Function: test_colorbar_errors(kwargs, error, message)

### Function: test_colorbar_axes_parmeters()

### Function: test_colorbar_wrong_figure()

### Function: test_colorbar_format_string_and_old()
