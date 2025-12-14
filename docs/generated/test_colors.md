## AI Summary

A file named test_colors.py.


### Function: test_create_lookup_table(N, result)

### Function: test_index_dtype(dtype)

### Function: test_resampled()

**Description:** GitHub issue #6025 pointed to incorrect ListedColormap.resampled;
here we test the method for LinearSegmentedColormap as well.

### Function: test_colormaps_get_cmap()

### Function: test_double_register_builtin_cmap()

### Function: test_colormap_copy()

### Function: test_colormap_equals()

### Function: test_colormap_endian()

**Description:** GitHub issue #1005: a bug in putmask caused erroneous
mapping of 1.0 when input from a non-native-byteorder
array.

### Function: test_colormap_invalid()

**Description:** GitHub issue #9892: Handling of nan's were getting mapped to under
rather than bad. This tests to make sure all invalid values
(-inf, nan, inf) are mapped respectively to (under, bad, over).

### Function: test_colormap_return_types()

**Description:** Make sure that tuples are returned for scalar input and
that the proper shapes are returned for ndarrays.

### Function: test_BoundaryNorm()

**Description:** GitHub issue #1258: interpolation was failing with numpy
1.7 pre-release.

### Function: test_CenteredNorm()

### Function: test_lognorm_invalid(vmin, vmax)

### Function: test_LogNorm()

**Description:** LogNorm ignored clip, now it has the same
behavior as Normalize, e.g., values > vmax are bigger than 1
without clip, with clip they are 1.

### Function: test_LogNorm_inverse()

**Description:** Test that lists work, and that the inverse works

### Function: test_PowerNorm()

### Function: test_PowerNorm_translation_invariance()

### Function: test_powernorm_cbar_limits()

### Function: test_Normalize()

### Function: test_FuncNorm()

### Function: test_TwoSlopeNorm_autoscale()

### Function: test_TwoSlopeNorm_autoscale_None_vmin()

### Function: test_TwoSlopeNorm_autoscale_None_vmax()

### Function: test_TwoSlopeNorm_scale()

### Function: test_TwoSlopeNorm_scaleout_center()

### Function: test_TwoSlopeNorm_scaleout_center_max()

### Function: test_TwoSlopeNorm_Even()

### Function: test_TwoSlopeNorm_Odd()

### Function: test_TwoSlopeNorm_VminEqualsVcenter()

### Function: test_TwoSlopeNorm_VmaxEqualsVcenter()

### Function: test_TwoSlopeNorm_VminGTVcenter()

### Function: test_TwoSlopeNorm_TwoSlopeNorm_VminGTVmax()

### Function: test_TwoSlopeNorm_VcenterGTVmax()

### Function: test_TwoSlopeNorm_premature_scaling()

### Function: test_SymLogNorm()

**Description:** Test SymLogNorm behavior

### Function: test_SymLogNorm_colorbar()

**Description:** Test un-called SymLogNorm in a colorbar.

### Function: test_SymLogNorm_single_zero()

**Description:** Test SymLogNorm to ensure it is not adding sub-ticks to zero label

## Class: TestAsinhNorm

**Description:** Tests for `~.colors.AsinhNorm`

### Function: _inverse_tester(norm_instance, vals)

**Description:** Checks if the inverse of the given normalization is working.

### Function: _scalar_tester(norm_instance, vals)

**Description:** Checks if scalars and arrays are handled the same way.
Tests only for float.

### Function: _mask_tester(norm_instance, vals)

**Description:** Checks mask handling

### Function: test_cmap_and_norm_from_levels_and_colors()

### Function: test_boundarynorm_and_colorbarbase()

### Function: test_cmap_and_norm_from_levels_and_colors2()

### Function: test_rgb_hsv_round_trip()

### Function: test_autoscale_masked()

### Function: test_light_source_topo_surface()

**Description:** Shades a DEM using different v.e.'s and blend modes.

### Function: test_light_source_shading_default()

**Description:** Array comparison test for the default "hsv" blend mode. Ensure the
default result doesn't change without warning.

### Function: test_light_source_shading_empty_mask()

### Function: test_light_source_masked_shading()

**Description:** Array comparison test for a surface with a masked portion. Ensures that
we don't wind up with "fringes" of odd colors around masked regions.

### Function: test_light_source_hillshading()

**Description:** Compare the current hillshading method against one that should be
mathematically equivalent. Illuminates a cone from a range of angles.

### Function: test_light_source_planar_hillshading()

**Description:** Ensure that the illumination intensity is correct for planar surfaces.

### Function: test_color_names()

### Function: _sph2cart(theta, phi)

### Function: _azimuth2math(azimuth, elevation)

**Description:** Convert from clockwise-from-north and up-from-horizontal to mathematical
conventions.

### Function: test_pandas_iterable(pd)

### Function: test_colormap_reversing(name)

**Description:** Check the generated _lut data of a colormap and corresponding reversed
colormap if they are almost the same.

### Function: test_has_alpha_channel()

### Function: test_cn()

### Function: test_conversions()

### Function: test_conversions_masked()

### Function: test_to_rgba_array_single_str()

### Function: test_to_rgba_array_2tuple_str()

### Function: test_to_rgba_array_alpha_array()

### Function: test_to_rgba_array_accepts_color_alpha_tuple()

### Function: test_to_rgba_array_explicit_alpha_overrides_tuple_alpha()

### Function: test_to_rgba_array_accepts_color_alpha_tuple_with_multiple_colors()

### Function: test_to_rgba_array_error_with_color_invalid_alpha_tuple()

### Function: test_to_rgba_accepts_color_alpha_tuple(rgba_alpha)

### Function: test_to_rgba_explicit_alpha_overrides_tuple_alpha()

### Function: test_to_rgba_error_with_color_invalid_alpha_tuple()

### Function: test_scalarmappable_to_rgba(bytes)

### Function: test_scalarmappable_nan_to_rgba(bytes)

### Function: test_failed_conversions()

### Function: test_grey_gray()

### Function: test_tableau_order()

### Function: test_ndarray_subclass_norm()

### Function: test_same_color()

### Function: test_hex_shorthand_notation()

### Function: test_repr_png()

### Function: test_repr_html()

### Function: test_get_under_over_bad()

### Function: test_non_mutable_get_values(kind)

### Function: test_colormap_alpha_array()

### Function: test_colormap_bad_data_with_alpha()

### Function: test_2d_to_rgba()

### Function: test_set_dict_to_rgba()

### Function: test_norm_deepcopy()

### Function: test_set_clim_emits_single_callback()

### Function: test_norm_callback()

### Function: test_scalarmappable_norm_update()

### Function: test_norm_update_figs(fig_test, fig_ref)

### Function: test_make_norm_from_scale_name()

### Function: test_color_sequences()

### Function: test_cm_set_cmap_error()

### Function: test_set_cmap_mismatched_name()

### Function: test_cmap_alias_names()

### Function: test_to_rgba_array_none_color_with_alpha_param()

### Function: test_is_color_like(input, expected)

### Function: test_colorizer_vmin_vmax()

### Function: forward(x)

### Function: inverse(x)

### Function: forward(x)

### Function: inverse(x)

### Function: test_init(self)

### Function: test_norm(self)

### Function: alternative_hillshade(azimuth, elev, z)

### Function: plane(azimuth, elevation, x, y)

**Description:** Create a plane whose normal vector is at the given azimuth and
elevation.

### Function: angled_plane(azimuth, elevation, angle, x, y)

**Description:** Create a plane whose normal vector is at an angle from the given
azimuth and elevation.

## Class: MyArray

### Function: __isub__(self, other)

### Function: __add__(self, other)
