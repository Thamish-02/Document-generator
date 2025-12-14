## AI Summary

A file named test_image.py.


### Function: test_alpha_interp()

**Description:** Test the interpolation of the alpha channel on RGBA images

### Function: test_interp_nearest_vs_none()

**Description:** Test the effect of "nearest" and "none" interpolation

### Function: test_figimage(suppressComposite)

### Function: test_image_python_io()

### Function: test_imshow_antialiased(fig_test, fig_ref, img_size, fig_size, interpolation)

### Function: test_imshow_zoom(fig_test, fig_ref)

### Function: test_imshow_pil(fig_test, fig_ref)

### Function: test_imread_pil_uint16()

### Function: test_imread_fspath()

### Function: test_imsave(fmt)

### Function: test_imsave_python_sequences()

### Function: test_imsave_rgba_origin(origin)

### Function: test_imsave_fspath(fmt)

### Function: test_imsave_color_alpha()

### Function: test_imsave_pil_kwargs_png()

### Function: test_imsave_pil_kwargs_tiff()

### Function: test_image_alpha()

### Function: test_imshow_alpha(fig_test, fig_ref)

### Function: test_cursor_data()

### Function: test_cursor_data_nonuniform(xy, data)

### Function: test_format_cursor_data(data, text)

### Function: test_image_clip()

### Function: test_image_cliprect()

### Function: test_imshow_10_10_1(fig_test, fig_ref)

### Function: test_imshow_10_10_2()

### Function: test_imshow_10_10_5()

### Function: test_no_interpolation_origin()

### Function: test_image_shift()

### Function: test_image_edges()

### Function: test_image_composite_background()

### Function: test_image_composite_alpha()

**Description:** Tests that the alpha value is recognized and correctly applied in the
process of compositing images together.

### Function: test_clip_path_disables_compositing(fig_test, fig_ref)

### Function: test_rasterize_dpi()

### Function: test_bbox_image_inverted()

### Function: test_get_window_extent_for_AxisImage()

### Function: test_zoom_and_clip_upper_origin()

### Function: test_nonuniformimage_setcmap()

### Function: test_nonuniformimage_setnorm()

### Function: test_jpeg_2d()

### Function: test_jpeg_alpha()

### Function: test_axesimage_setdata()

### Function: test_figureimage_setdata()

### Function: test_setdata_xya(image_cls, x, y, a)

### Function: test_minimized_rasterized()

### Function: test_load_from_url()

### Function: test_log_scale_image()

### Function: test_rotate_image()

### Function: test_image_preserve_size()

### Function: test_image_preserve_size2()

### Function: test_mask_image_over_under()

### Function: test_mask_image()

### Function: test_mask_image_all()

### Function: test_imshow_endianess()

### Function: test_imshow_masked_interpolation()

### Function: test_imshow_no_warn_invalid()

### Function: test_imshow_clips_rgb_to_valid_range(dtype)

### Function: test_imshow_flatfield()

### Function: test_imshow_bignumbers()

### Function: test_imshow_bignumbers_real()

### Function: test_empty_imshow(make_norm)

### Function: test_imshow_float16()

### Function: test_imshow_float128()

### Function: test_imshow_bool()

### Function: test_full_invalid()

### Function: test_composite(fmt, counted, composite_image, count)

### Function: test_relim()

### Function: test_unclipped()

### Function: test_respects_bbox()

### Function: test_image_cursor_formatting()

### Function: test_image_array_alpha(fig_test, fig_ref)

**Description:** Per-pixel alpha channel test.

### Function: test_image_array_alpha_validation()

### Function: test_exact_vmin()

### Function: test_image_placement()

**Description:** The red box should line up exactly with the outside of the image.

## Class: QuantityND

### Function: test_quantitynd()

### Function: test_imshow_quantitynd()

### Function: test_norm_change(fig_test, fig_ref)

### Function: test_huge_range_log(fig_test, fig_ref, x)

### Function: test_spy_box(fig_test, fig_ref)

### Function: test_nonuniform_and_pcolor()

### Function: test_nonuniform_logscale()

### Function: test_rgba_antialias()

### Function: test_upsample_interpolation_stage(fig_test, fig_ref)

**Description:** Show that interpolation_stage='auto' gives the same as 'data'
for upsampling.

### Function: test_downsample_interpolation_stage(fig_test, fig_ref)

**Description:** Show that interpolation_stage='auto' gives the same as 'rgba'
for downsampling.

### Function: test_rc_interpolation_stage()

### Function: test_large_image(fig_test, fig_ref, dim, size, msg, origin)

### Function: test_str_norms(fig_test, fig_ref)

### Function: test__resample_valid_output()

### Function: test_axesimage_get_shape()

### Function: test_non_transdata_image_does_not_touch_aspect()

### Function: test_downsampling()

### Function: test_downsampling_speckle()

### Function: test_upsampling()

### Function: test_resample_dtypes(dtype, ndim)

### Function: test_interpolation_stage_rgba_respects_alpha_param(fig_test, fig_ref, intp_stage)

### Function: __new__(cls, input_array, units)

### Function: __array_finalize__(self, obj)

### Function: __getitem__(self, item)

### Function: __array_ufunc__(self, ufunc, method)

### Function: v(self)
