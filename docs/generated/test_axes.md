## AI Summary

A file named test_axes.py.


### Function: test_invisible_axes(fig_test, fig_ref)

### Function: test_get_labels()

### Function: test_repr()

### Function: test_label_loc_vertical(fig_test, fig_ref)

### Function: test_label_loc_horizontal(fig_test, fig_ref)

### Function: test_label_loc_rc(fig_test, fig_ref)

### Function: test_label_shift()

### Function: test_acorr(fig_test, fig_ref)

### Function: test_acorr_integers(fig_test, fig_ref)

### Function: test_spy(fig_test, fig_ref)

### Function: test_spy_invalid_kwargs()

### Function: test_matshow(fig_test, fig_ref)

### Function: test_formatter_ticker()

### Function: test_funcformatter_auto_formatter()

### Function: test_strmethodformatter_auto_formatter()

### Function: test_twin_axis_locators_formatters()

### Function: test_twinx_cla()

### Function: test_twin_units(twin)

### Function: test_twin_logscale(fig_test, fig_ref, twin)

### Function: test_twinx_axis_scales()

### Function: test_twin_inherit_autoscale_setting()

### Function: test_inverted_cla()

### Function: test_subclass_clear_cla()

### Function: test_cla_not_redefined_internally()

### Function: test_minorticks_on_rcParams_both(fig_test, fig_ref)

### Function: test_autoscale_tiny_range()

### Function: test_autoscale_tight()

### Function: test_autoscale_log_shared()

### Function: test_use_sticky_edges()

### Function: test_sticky_shared_axes(fig_test, fig_ref)

### Function: test_sticky_tolerance()

### Function: test_sticky_tolerance_contourf()

### Function: test_nargs_stem()

### Function: test_nargs_legend()

### Function: test_nargs_pcolorfast()

### Function: test_basic_annotate()

### Function: test_arrow_simple()

### Function: test_arrow_empty()

### Function: test_arrow_in_view()

### Function: test_annotate_default_arrow()

### Function: test_annotate_signature()

**Description:** Check that the signature of Axes.annotate() matches Annotation.

### Function: test_fill_units()

### Function: test_plot_format_kwarg_redundant()

### Function: test_errorbar_dashes(fig_test, fig_ref)

### Function: test_errorbar_mapview_kwarg()

### Function: test_single_point()

### Function: test_single_date()

### Function: test_shaped_data(fig_test, fig_ref)

### Function: test_structured_data()

### Function: test_aitoff_proj()

**Description:** Test aitoff projection ref.:
https://github.com/matplotlib/matplotlib/pull/14451

### Function: test_axvspan_epoch()

### Function: test_axhspan_epoch()

### Function: test_hexbin_extent()

### Function: test_hexbin_bad_extents()

### Function: test_hexbin_string_norm()

### Function: test_hexbin_empty()

### Function: test_hexbin_pickable()

### Function: test_hexbin_log()

### Function: test_hexbin_log_offsets()

### Function: test_hexbin_linear()

### Function: test_hexbin_log_clim()

### Function: test_hexbin_mincnt_behavior_upon_C_parameter(fig_test, fig_ref)

### Function: test_inverted_limits()

### Function: test_nonfinite_limits()

### Function: test_limits_empty_data(plot_fun, fig_test, fig_ref)

### Function: test_imshow()

### Function: test_imshow_clip()

### Function: test_imshow_norm_vminvmax()

**Description:** Parameters vmin, vmax should error if norm is given.

### Function: test_polycollection_joinstyle()

### Function: test_fill_between_input(x, y1, y2)

### Function: test_fill_betweenx_input(y, x1, x2)

### Function: test_fill_between_interpolate()

### Function: test_fill_between_interpolate_decreasing()

### Function: test_fill_between_interpolate_nan()

### Function: test_symlog()

### Function: test_symlog2()

### Function: test_pcolorargs_5205()

### Function: test_pcolormesh()

### Function: test_pcolormesh_small()

### Function: test_pcolormesh_alpha()

### Function: test_pcolormesh_rgba(fig_test, fig_ref, dims, alpha)

### Function: test_pcolormesh_nearest_noargs(fig_test, fig_ref)

### Function: test_pcolormesh_datetime_axis()

### Function: test_pcolor_datetime_axis()

### Function: test_pcolorargs()

### Function: test_pcolormesh_underflow_error()

**Description:** Test that underflow errors don't crop up in pcolormesh.  Probably
a numpy bug (https://github.com/numpy/numpy/issues/25810).

### Function: test_pcolorargs_with_read_only()

### Function: test_pcolornearest(fig_test, fig_ref)

### Function: test_pcolornearestunits(fig_test, fig_ref)

### Function: test_pcolorflaterror()

### Function: test_samesizepcolorflaterror()

### Function: test_pcolorauto(fig_test, fig_ref, snap)

### Function: test_canonical()

### Function: test_arc_angles()

### Function: test_arc_ellipse()

### Function: test_marker_as_markerstyle()

### Function: test_markevery()

### Function: test_markevery_line()

### Function: test_markevery_linear_scales()

### Function: test_markevery_linear_scales_zoomed()

### Function: test_markevery_log_scales()

### Function: test_markevery_polar()

### Function: test_markevery_linear_scales_nans()

### Function: test_marker_edges()

### Function: test_bar_tick_label_single()

### Function: test_nan_bar_values()

### Function: test_bar_ticklabel_fail()

### Function: test_bar_tick_label_multiple()

### Function: test_bar_tick_label_multiple_old_alignment()

### Function: test_bar_decimal_center(fig_test, fig_ref)

### Function: test_barh_decimal_center(fig_test, fig_ref)

### Function: test_bar_decimal_width(fig_test, fig_ref)

### Function: test_barh_decimal_height(fig_test, fig_ref)

### Function: test_bar_color_none_alpha()

### Function: test_bar_edgecolor_none_alpha()

### Function: test_barh_tick_label()

### Function: test_bar_timedelta()

**Description:** Smoketest that bar can handle width and height in delta units.

### Function: test_bar_datetime_start()

**Description:** test that tickers are correct for datetimes

### Function: test_boxplot_dates_pandas(pd)

### Function: test_boxplot_capwidths()

### Function: test_pcolor_regression(pd)

### Function: test_bar_pandas(pd)

### Function: test_bar_pandas_indexed(pd)

### Function: test_bar_hatches(fig_test, fig_ref)

### Function: test_bar_labels(x, width, label, expected_labels, container_label)

### Function: test_bar_labels_length()

### Function: test_pandas_minimal_plot(pd)

### Function: test_hist_log()

### Function: test_hist_log_2(fig_test, fig_ref)

### Function: test_hist_log_barstacked()

### Function: test_hist_bar_empty()

### Function: test_hist_float16()

### Function: test_hist_step_empty()

### Function: test_hist_step_filled()

### Function: test_hist_density()

### Function: test_hist_unequal_bins_density()

### Function: test_hist_datetime_datasets()

### Function: test_hist_datetime_datasets_bins(bins_preprocess)

### Function: test_hist_with_empty_input(data, expected_number_of_hists)

### Function: test_hist_zorder(histtype, zorder)

### Function: test_stairs_no_baseline_fill_warns()

### Function: test_stairs(fig_test, fig_ref)

### Function: test_stairs_fill(fig_test, fig_ref)

### Function: test_stairs_update(fig_test, fig_ref)

### Function: test_stairs_baseline_None(fig_test, fig_ref)

### Function: test_stairs_empty()

### Function: test_stairs_invalid_nan()

### Function: test_stairs_invalid_mismatch()

### Function: test_stairs_invalid_update()

### Function: test_stairs_invalid_update2()

### Function: test_stairs_options()

### Function: test_stairs_datetime()

### Function: test_stairs_edge_handling(fig_test, fig_ref)

### Function: contour_dat()

### Function: test_contour_hatching()

### Function: test_contour_colorbar()

### Function: test_hist2d()

### Function: test_hist2d_transpose()

### Function: test_hist2d_density()

## Class: TestScatter

### Function: _params(c, xsize)

### Function: test_parse_scatter_color_args(params, expected_result)

### Function: test_parse_scatter_color_args_edgecolors(kwargs, expected_edgecolors)

### Function: test_parse_scatter_color_args_error()

### Function: test_parse_c_facecolor_warning_direct(c, facecolor)

**Description:** Test the internal _parse_scatter_color_args method directly.

### Function: test_scatter_c_facecolor_warning_integration(c, facecolor)

**Description:** Test the warning through the actual scatter plot creation.

### Function: test_as_mpl_axes_api()

### Function: test_pyplot_axes()

### Function: test_log_scales()

### Function: test_log_scales_no_data()

### Function: test_log_scales_invalid()

### Function: test_stackplot()

### Function: test_stackplot_baseline()

### Function: test_stackplot_hatching(fig_ref, fig_test)

### Function: test_stackplot_subfig_legend()

### Function: _bxp_test_helper(stats_kwargs, transform_stats, bxp_kwargs)

### Function: test_bxp_baseline()

### Function: test_bxp_rangewhis()

### Function: test_bxp_percentilewhis()

### Function: test_bxp_with_xlabels()

### Function: test_bxp_horizontal()

### Function: test_bxp_with_ylabels()

### Function: test_bxp_patchartist()

### Function: test_bxp_custompatchartist()

### Function: test_bxp_customoutlier()

### Function: test_bxp_showcustommean()

### Function: test_bxp_custombox()

### Function: test_bxp_custommedian()

### Function: test_bxp_customcap()

### Function: test_bxp_customwhisker()

### Function: test_boxplot_median_bound_by_box(fig_test, fig_ref)

### Function: test_bxp_shownotches()

### Function: test_bxp_nocaps()

### Function: test_bxp_nobox()

### Function: test_bxp_no_flier_stats()

### Function: test_bxp_showmean()

### Function: test_bxp_showmeanasline()

### Function: test_bxp_scalarwidth()

### Function: test_bxp_customwidths()

### Function: test_bxp_custompositions()

### Function: test_bxp_bad_widths()

### Function: test_bxp_bad_positions()

### Function: test_bxp_custom_capwidths()

### Function: test_bxp_custom_capwidth()

### Function: test_bxp_bad_capwidths()

### Function: test_boxplot()

### Function: test_boxplot_masked(fig_test, fig_ref)

### Function: test_boxplot_custom_capwidths()

### Function: test_boxplot_sym2()

### Function: test_boxplot_sym()

### Function: test_boxplot_autorange_whiskers()

### Function: _rc_test_bxp_helper(ax, rc_dict)

### Function: test_boxplot_rc_parameters()

### Function: test_boxplot_with_CIarray()

### Function: test_boxplot_no_weird_whisker()

### Function: test_boxplot_bad_medians()

### Function: test_boxplot_bad_ci()

### Function: test_boxplot_zorder()

### Function: test_boxplot_marker_behavior()

### Function: test_boxplot_mod_artist_after_plotting()

### Function: test_vert_violinplot_baseline()

### Function: test_vert_violinplot_showmeans()

### Function: test_vert_violinplot_showextrema()

### Function: test_vert_violinplot_showmedians()

### Function: test_vert_violinplot_showall()

### Function: test_vert_violinplot_custompoints_10()

### Function: test_vert_violinplot_custompoints_200()

### Function: test_horiz_violinplot_baseline()

### Function: test_horiz_violinplot_showmedians()

### Function: test_horiz_violinplot_showmeans()

### Function: test_horiz_violinplot_showextrema()

### Function: test_horiz_violinplot_showall()

### Function: test_horiz_violinplot_custompoints_10()

### Function: test_horiz_violinplot_custompoints_200()

### Function: test_violinplot_sides()

### Function: test_violinplot_bad_positions()

### Function: test_violinplot_bad_widths()

### Function: test_violinplot_bad_quantiles()

### Function: test_violinplot_outofrange_quantiles()

### Function: test_violinplot_single_list_quantiles(fig_test, fig_ref)

### Function: test_violinplot_pandas_series(fig_test, fig_ref, pd)

### Function: test_manage_xticks()

### Function: test_boxplot_not_single()

### Function: test_tick_space_size_0()

### Function: test_errorbar()

### Function: test_mixed_errorbar_polar_caps()

**Description:** Mix several polar errorbar use cases in a single test figure.

It is advisable to position individual points off the grid. If there are
problems with reproducibility of this test, consider removing grid.

### Function: test_errorbar_colorcycle()

### Function: test_errorbar_cycle_ecolor(fig_test, fig_ref)

### Function: test_errorbar_shape()

### Function: test_errorbar_limits()

### Function: test_errorbar_nonefmt()

### Function: test_errorbar_remove()

### Function: test_errorbar_line_specific_kwargs()

### Function: test_errorbar_with_prop_cycle(fig_test, fig_ref)

### Function: test_errorbar_every_invalid()

### Function: test_xerr_yerr_not_negative()

### Function: test_xerr_yerr_not_none()

### Function: test_errorbar_every(fig_test, fig_ref)

### Function: test_errorbar_linewidth_type(elinewidth)

### Function: test_errorbar_nan(fig_test, fig_ref)

### Function: test_errorbar_masked_negative(fig_test, fig_ref)

### Function: test_hist_stacked_stepfilled()

### Function: test_hist_offset()

### Function: test_hist_step()

### Function: test_hist_step_horiz()

### Function: test_hist_stacked_weighted()

### Function: test_stem()

### Function: test_stem_args()

**Description:** Test that stem() correctly identifies x and y values.

### Function: test_stem_markerfmt()

**Description:** Test that stem(..., markerfmt=...) produces the intended markers.

### Function: test_stem_dates()

### Function: test_stem_orientation()

### Function: test_hist_stacked_stepfilled_alpha()

### Function: test_hist_stacked_step()

### Function: test_hist_stacked_density()

### Function: test_hist_step_bottom()

### Function: test_hist_step_geometry()

### Function: test_hist_step_bottom_geometry()

### Function: test_hist_stacked_step_geometry()

### Function: test_hist_stacked_step_bottom_geometry()

### Function: test_hist_stacked_bar()

### Function: test_hist_vectorized_params(fig_test, fig_ref, kwargs)

### Function: test_hist_sequence_type_styles()

### Function: test_hist_color_none()

### Function: test_hist_color_semantics(kwargs, patch_face, patch_edge)

### Function: test_hist_barstacked_bottom_unchanged()

### Function: test_hist_emptydata()

### Function: test_hist_unused_labels()

### Function: test_hist_labels()

### Function: test_transparent_markers()

### Function: test_rgba_markers()

### Function: test_mollweide_grid()

### Function: test_mollweide_forward_inverse_closure()

### Function: test_mollweide_inverse_forward_closure()

### Function: test_alpha()

### Function: test_eventplot()

### Function: test_eventplot_defaults()

**Description:** test that eventplot produces the correct output given the default params
(see bug #3728)

### Function: test_eventplot_colors(colors)

**Description:** Test the *colors* parameter of eventplot. Inspired by issue #8193.

### Function: test_eventplot_alpha()

### Function: test_eventplot_problem_kwargs(recwarn)

**Description:** test that 'singular' versions of LineCollection props raise an
MatplotlibDeprecationWarning rather than overriding the 'plural' versions
(e.g., to prevent 'color' from overriding 'colors', see issue #4297)

### Function: test_empty_eventplot()

### Function: test_eventplot_orientation(data, orientation)

**Description:** Introduced when fixing issue #6412.

### Function: test_eventplot_units_list(fig_test, fig_ref)

### Function: test_marker_styles()

### Function: test_markers_fillstyle_rcparams()

### Function: test_vertex_markers()

### Function: test_eb_line_zorder()

### Function: test_axline_loglog(fig_test, fig_ref)

### Function: test_axline(fig_test, fig_ref)

### Function: test_axline_transaxes(fig_test, fig_ref)

### Function: test_axline_transaxes_panzoom(fig_test, fig_ref)

### Function: test_axline_args()

**Description:** Exactly one of *xy2* and *slope* must be specified.

### Function: test_vlines()

### Function: test_vlines_default()

### Function: test_hlines()

### Function: test_hlines_default()

### Function: test_lines_with_colors(fig_test, fig_ref, data)

### Function: test_vlines_hlines_blended_transform()

### Function: test_step_linestyle()

### Function: test_mixed_collection()

### Function: test_subplot_key_hash()

### Function: test_specgram()

**Description:** Test axes.specgram in default (psd) mode.

### Function: test_specgram_magnitude()

**Description:** Test axes.specgram in magnitude mode.

### Function: test_specgram_angle()

**Description:** Test axes.specgram in angle and phase modes.

### Function: test_specgram_fs_none()

**Description:** Test axes.specgram when Fs is None, should not throw error.

### Function: test_specgram_origin_rcparam(fig_test, fig_ref)

**Description:** Test specgram ignores image.origin rcParam and uses origin 'upper'.

### Function: test_specgram_origin_kwarg()

**Description:** Ensure passing origin as a kwarg raises a TypeError.

### Function: test_psd_csd()

### Function: test_spectrum()

### Function: test_psd_csd_edge_cases()

### Function: test_twin_remove(fig_test, fig_ref)

### Function: test_twin_spines()

### Function: test_twin_spines_on_top()

### Function: test_rcparam_grid_minor(grid_which, major_visible, minor_visible)

### Function: test_grid()

### Function: test_reset_grid()

### Function: test_reset_ticks(fig_test, fig_ref)

### Function: test_context_ticks()

### Function: test_vline_limit()

### Function: test_axline_minmax(fv, fh, args)

### Function: test_empty_shared_subplots()

### Function: test_shared_with_aspect_1()

### Function: test_shared_with_aspect_2()

### Function: test_shared_with_aspect_3()

### Function: test_shared_aspect_error()

### Function: test_axis_errors(err, args, kwargs, match)

### Function: test_axis_method_errors()

### Function: test_twin_with_aspect(twin)

### Function: test_relim_visible_only()

### Function: test_text_labelsize()

**Description:** tests for issue #1172

### Function: test_pie_default()

### Function: test_pie_linewidth_0()

### Function: test_pie_center_radius()

### Function: test_pie_linewidth_2()

### Function: test_pie_ccw_true()

### Function: test_pie_frame_grid()

### Function: test_pie_rotatelabels_true()

### Function: test_pie_nolabel_but_legend()

### Function: test_pie_shadow()

### Function: test_pie_textprops()

### Function: test_pie_get_negative_values()

### Function: test_pie_invalid_explode()

### Function: test_pie_invalid_labels()

### Function: test_pie_invalid_radius()

### Function: test_normalize_kwarg_pie()

### Function: test_pie_hatch_single(fig_test, fig_ref)

### Function: test_pie_hatch_multi(fig_test, fig_ref)

### Function: test_set_get_ticklabels()

### Function: test_set_ticks_kwargs_raise_error_without_labels()

**Description:** When labels=None and any kwarg is passed, axis.set_ticks() raises a
ValueError.

### Function: test_set_ticks_with_labels(fig_test, fig_ref)

**Description:** Test that these two are identical::

    set_xticks(ticks); set_xticklabels(labels, **kwargs)
    set_xticks(ticks, labels, **kwargs)

### Function: test_xticks_bad_args()

### Function: test_subsampled_ticklabels()

### Function: test_mismatched_ticklabels()

### Function: test_empty_ticks_fixed_loc()

### Function: test_retain_tick_visibility()

### Function: test_warn_too_few_labels()

### Function: test_tick_label_update()

### Function: test_o_marker_path_snap()

### Function: test_margins()

### Function: test_margin_getters()

### Function: test_set_margin_updates_limits()

### Function: test_margins_errors(err, args, kwargs, match)

### Function: test_length_one_hist()

### Function: test_set_xy_bound()

### Function: test_pathological_hexbin()

### Function: test_color_None()

### Function: test_color_alias()

### Function: test_numerical_hist_label()

### Function: test_unicode_hist_label()

### Function: test_move_offsetlabel()

### Function: test_rc_spines()

### Function: test_rc_grid()

### Function: test_rc_tick()

### Function: test_rc_major_minor_tick()

### Function: test_square_plot()

### Function: test_bad_plot_args()

### Function: test_pcolorfast(xy, data, cls)

### Function: test_pcolorfast_bad_dims()

### Function: test_pcolorfast_regular_xy_incompatible_size()

**Description:** Test that the sizes of X, Y, C are compatible for regularly spaced X, Y.

Note that after the regualar-spacing check, pcolorfast may go into the
fast "image" mode, where the individual X, Y positions are not used anymore.
Therefore, the algorithm had worked with any regularly number of regularly
spaced values, but discarded their values.

### Function: test_shared_scale()

### Function: test_shared_bool()

### Function: test_violin_point_mass()

**Description:** Violin plot should handle point mass pdf gracefully.

### Function: generate_errorbar_inputs()

### Function: test_errorbar_inputs_shotgun(kwargs)

### Function: test_dash_offset()

### Function: test_title_pad()

### Function: test_title_location_roundtrip()

### Function: test_title_location_shared(sharex)

### Function: test_loglog()

### Function: test_loglog_nonpos()

### Function: test_axes_margins()

### Function: shared_axis_remover(request)

### Function: shared_axes_generator(request)

### Function: test_remove_shared_axes(shared_axes_generator, shared_axis_remover)

### Function: test_remove_shared_axes_relim()

### Function: test_shared_axes_autoscale()

### Function: test_adjust_numtick_aspect()

### Function: test_auto_numticks()

### Function: test_auto_numticks_log()

### Function: test_broken_barh_empty()

### Function: test_broken_barh_timedelta()

**Description:** Check that timedelta works as x, dx pair for this method.

### Function: test_pandas_pcolormesh(pd)

### Function: test_pandas_indexing_dates(pd)

### Function: test_pandas_errorbar_indexing(pd)

### Function: test_pandas_index_shape(pd)

### Function: test_pandas_indexing_hist(pd)

### Function: test_pandas_bar_align_center(pd)

### Function: test_axis_get_tick_params()

### Function: test_axis_set_tick_params_labelsize_labelcolor()

### Function: test_axes_tick_params_gridlines()

### Function: test_axes_tick_params_ylabelside()

### Function: test_axes_tick_params_xlabelside()

### Function: test_none_kwargs()

### Function: test_bar_uint8()

### Function: test_date_timezone_x()

### Function: test_date_timezone_y()

### Function: test_date_timezone_x_and_y()

### Function: test_axisbelow()

### Function: test_titletwiny()

### Function: test_titlesetpos()

### Function: test_title_xticks_top()

### Function: test_title_xticks_top_both()

### Function: test_title_above_offset(left, center)

### Function: test_title_no_move_off_page()

### Function: test_title_inset_ax()

### Function: test_offset_label_color()

### Function: test_offset_text_visible()

### Function: test_large_offset()

### Function: test_barb_units()

### Function: test_quiver_units()

### Function: test_bar_color_cycle()

### Function: test_tick_param_label_rotation()

### Function: test_fillbetween_cycle()

### Function: test_log_margins()

### Function: test_color_length_mismatch()

### Function: test_eventplot_legend()

### Function: test_eventplot_errors(err, args, kwargs, match)

### Function: test_bar_broadcast_args()

### Function: test_invalid_axis_limits()

### Function: test_minorticks_on(xscale, yscale)

### Function: test_twinx_knows_limits()

### Function: test_zero_linewidth()

### Function: test_empty_errorbar_legend()

### Function: test_plot_decimal(fig_test, fig_ref)

### Function: test_markerfacecolor_none_alpha(fig_test, fig_ref)

### Function: test_tick_padding_tightbbox()

**Description:** Test that tick padding gets turned off if axis is off

### Function: test_inset()

**Description:** Ensure that inset_ax argument is indeed optional

### Function: test_zoom_inset()

### Function: test_inset_polar()

### Function: test_inset_projection()

### Function: test_inset_subclass()

### Function: test_indicate_inset_inverted(x_inverted, y_inverted)

**Description:** Test that the inset lines are correctly located with inverted data axes.

### Function: test_set_position()

### Function: test_spines_properbbox_after_zoom()

### Function: test_limits_after_scroll_zoom()

### Function: test_gettightbbox_ignore_nan()

### Function: test_scatter_series_non_zero_index(pd)

### Function: test_scatter_empty_data()

### Function: test_annotate_across_transforms()

## Class: _Translation

### Function: test_secondary_xy()

### Function: test_secondary_fail()

### Function: test_secondary_resize()

### Function: test_secondary_minorloc()

### Function: test_secondary_formatter()

### Function: test_secondary_repr()

### Function: test_axis_options()

### Function: color_boxes(fig, ax)

**Description:** Helper for the tests below that test the extents of various Axes elements

### Function: test_normal_axes()

### Function: test_nodecorator()

### Function: test_displaced_spine()

### Function: test_tickdirs()

**Description:** Switch the tickdirs and make sure the bboxes switch with them

### Function: test_minor_accountedfor()

### Function: test_axis_bool_arguments(fig_test, fig_ref)

### Function: test_axis_extent_arg()

### Function: test_axis_extent_arg2()

### Function: test_hist_auto_bins()

### Function: test_hist_nan_data()

### Function: test_hist_range_and_density()

### Function: test_bar_errbar_zorder()

### Function: test_set_ticks_inverted()

### Function: test_aspect_nonlinear_adjustable_box()

### Function: test_aspect_nonlinear_adjustable_datalim()

### Function: test_box_aspect()

### Function: test_box_aspect_custom_position()

### Function: test_bbox_aspect_axes_init()

### Function: test_set_aspect_negative()

### Function: test_redraw_in_frame()

### Function: test_invisible_axes_events()

### Function: test_xtickcolor_is_not_markercolor()

### Function: test_ytickcolor_is_not_markercolor()

### Function: test_unautoscale(axis, auto)

### Function: test_polar_interpolation_steps_variable_r(fig_test, fig_ref)

### Function: test_autoscale_tiny_sticky()

### Function: test_xtickcolor_is_not_xticklabelcolor()

### Function: test_ytickcolor_is_not_yticklabelcolor()

### Function: test_xaxis_offsetText_color()

### Function: test_yaxis_offsetText_color()

### Function: test_relative_ticklabel_sizes(size)

### Function: test_multiplot_autoscale()

### Function: test_sharing_does_not_link_positions()

### Function: test_2dcolor_plot(fig_test, fig_ref)

### Function: test_shared_axes_clear(fig_test, fig_ref)

### Function: test_shared_axes_retick()

### Function: test_ylabel_ha_with_position(ha)

### Function: test_bar_label_location_vertical()

### Function: test_bar_label_location_vertical_yinverted()

### Function: test_bar_label_location_horizontal()

### Function: test_bar_label_location_horizontal_yinverted()

### Function: test_bar_label_location_horizontal_xinverted()

### Function: test_bar_label_location_horizontal_xyinverted()

### Function: test_bar_label_location_center()

### Function: test_centered_bar_label_nonlinear()

### Function: test_centered_bar_label_label_beyond_limits()

### Function: test_bar_label_location_errorbars()

### Function: test_bar_label_fmt(fmt)

### Function: test_bar_label_fmt_error()

### Function: test_bar_label_labels()

### Function: test_bar_label_nan_ydata()

### Function: test_bar_label_nan_ydata_inverted()

### Function: test_nan_barlabels()

### Function: test_patch_bounds()

### Function: test_warn_ignored_scatter_kwargs()

### Function: test_artist_sublists()

### Function: test_empty_line_plots()

### Function: test_plot_format_errors(fmt, match, data)

### Function: test_plot_format()

### Function: test_automatic_legend()

### Function: test_plot_errors()

### Function: test_clim()

### Function: test_bezier_autoscale()

### Function: test_small_autoscale()

### Function: test_get_xticklabel()

### Function: test_bar_leading_nan()

### Function: test_bar_all_nan(fig_test, fig_ref)

### Function: test_extent_units()

### Function: test_cla_clears_children_axes_and_fig()

### Function: test_child_axes_removal()

### Function: test_scatter_color_repr_error()

### Function: test_zorder_and_explicit_rasterization()

### Function: test_preset_clip_paths()

### Function: test_rc_axes_label_formatting()

### Function: test_ecdf(fig_test, fig_ref)

### Function: test_ecdf_invalid()

### Function: test_fill_between_axes_limits()

### Function: test_tick_param_labelfont()

### Function: test_set_secondary_axis_color()

### Function: test_xylim_changed_shared()

### Function: test_axhvlinespan_interpolation()

### Function: test_axes_clear_behavior(fig_ref, fig_test, which)

**Description:** Test that the given tick params are not reset by ax.clear().

### Function: test_axes_clear_reference_cycle()

### Function: test_boxplot_tick_labels()

### Function: test_latex_pie_percent(fig_test, fig_ref)

### Function: test_violinplot_orientation(fig_test, fig_ref)

### Function: test_boxplot_orientation(fig_test, fig_ref)

### Function: test_use_colorizer_keyword()

### Function: test_wrong_use_colorizer()

### Function: test_bar_color_precedence()

### Function: test_axes_set_position_external_bbox_unchanged(fig_test, fig_ref)

### Function: test_caps_color()

### Function: test_caps_no_ecolor()

### Function: _formfunc(x, pos)

## Class: SubClaAxes

## Class: ClearAxes

## Class: ClearSuperAxes

## Class: SubClearAxes

### Function: test_scatter_plot(self)

### Function: test_scatter_marker(self)

### Function: test_scatter_2D(self)

### Function: test_scatter_decimal(self, fig_test, fig_ref)

### Function: test_scatter_color(self)

### Function: test_scatter_color_warning(self, kwargs)

### Function: test_scatter_unfilled(self)

### Function: test_scatter_unfillable(self)

### Function: test_scatter_size_arg_size(self)

### Function: test_scatter_edgecolor_RGB(self)

### Function: test_scatter_invalid_color(self, fig_test, fig_ref)

### Function: test_scatter_no_invalid_color(self, fig_test, fig_ref)

### Function: test_scatter_norm_vminvmax(self)

**Description:** Parameters vmin, vmax should error if norm is given.

### Function: test_scatter_single_point(self, fig_test, fig_ref)

### Function: test_scatter_different_shapes(self, fig_test, fig_ref)

### Function: test_scatter_c(self, c_case, re_key)

### Function: test_scatter_single_color_c(self, fig_test, fig_ref)

### Function: test_scatter_linewidths(self)

### Function: test_scatter_singular_plural_arguments(self)

### Function: get_next_color()

### Function: get_next_color()

### Function: get_next_color()

### Function: get_next_color()

## Class: Polar

### Function: layers(n, m)

### Function: transform(stats)

### Function: transform(stats)

### Function: transform(stats)

### Function: _assert_equal(stem_container, expected)

### Function: _assert_equal(stem_container, linecolor, markercolor, marker)

**Description:** Check that the given StemContainer has the properties listed as
keyword-arguments.

### Function: make_patch_spines_invisible(ax)

### Function: formatter_func(x, pos)

### Function: _helper_x(ax)

### Function: _helper_y(ax)

### Function: __init__(self, dx)

### Function: transform(self, values)

### Function: inverted(self)

### Function: invert(x)

### Function: invert(x)

### Function: invert(x)

### Function: get_next_color()

### Function: assert_not_in_reference_cycle(start)

## Class: ClaAxes

## Class: ClaSuperAxes

### Function: clear(self)

### Function: clear(self)

### Function: get_next_color()

### Function: __init__(self)

### Function: _as_mpl_axes(self)

### Function: cla(self)

### Function: cla(self)
