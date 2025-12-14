## AI Summary

A file named test_transforms.py.


## Class: TestAffine2D

## Class: TestAffineDeltaTransform

### Function: test_non_affine_caching()

### Function: test_external_transform_api()

### Function: test_pre_transform_plotting()

### Function: test_contour_pre_transform_limits()

### Function: test_pcolor_pre_transform_limits()

### Function: test_pcolormesh_pre_transform_limits()

### Function: test_pcolormesh_gouraud_nans()

### Function: test_Affine2D_from_values()

### Function: test_affine_inverted_invalidated()

### Function: test_clipping_of_log()

## Class: NonAffineForTest

**Description:** A class which looks like a non affine transform, but does whatever
the given transform does (even if it is affine). This is very useful
for testing NonAffine behaviour with a simple Affine transform.

## Class: TestBasicTransform

## Class: TestTransformPlotInterface

### Function: assert_bbox_eq(bbox1, bbox2)

### Function: test_bbox_frozen_copies_minpos()

### Function: test_bbox_intersection()

### Function: test_bbox_as_strings()

### Function: test_str_transform()

### Function: test_transform_single_point()

### Function: test_log_transform()

### Function: test_nan_overlap()

### Function: test_transform_angles()

### Function: test_nonsingular()

### Function: test_transformed_path()

### Function: test_transformed_patch_path()

### Function: test_lockable_bbox(locked_element)

### Function: test_transformwrapper()

### Function: test_scale_swapping(fig_test, fig_ref)

### Function: test_offset_copy_errors()

### Function: test_transformedbbox_contains()

### Function: test_interval_contains()

### Function: test_interval_contains_open()

### Function: test_scaledrotation_initialization()

**Description:** Test that the ScaledRotation object is initialized correctly.

### Function: test_scaledrotation_get_matrix_invalid()

**Description:** Test get_matrix when the matrix is invalid and needs recalculation.

### Function: test_init(self)

### Function: test_values(self)

### Function: test_modify_inplace(self)

### Function: test_clear(self)

### Function: test_rotate(self)

### Function: test_rotate_around(self)

### Function: test_scale(self)

### Function: test_skew(self)

### Function: test_translate(self)

### Function: test_rotate_plus_other(self)

### Function: test_rotate_around_plus_other(self)

### Function: test_scale_plus_other(self)

### Function: test_skew_plus_other(self)

### Function: test_translate_plus_other(self)

### Function: test_invalid_transform(self)

### Function: test_copy(self)

### Function: test_deepcopy(self)

### Function: test_invalidate(self)

## Class: AssertingNonAffineTransform

**Description:** This transform raises an assertion error when called when it
shouldn't be and ``self.raise_on_transform`` is True.

## Class: ScaledBy

### Function: __init__(self, real_trans)

### Function: transform_non_affine(self, values)

### Function: transform_path_non_affine(self, path)

### Function: setup_method(self)

### Function: test_transform_depth(self)

### Function: test_left_to_right_iteration(self)

### Function: test_transform_shortcuts(self)

### Function: test_contains_branch(self)

### Function: test_affine_simplification(self)

### Function: test_line_extent_axes_coords(self)

### Function: test_line_extent_data_coords(self)

### Function: test_line_extent_compound_coords1(self)

### Function: test_line_extent_predata_transform_coords(self)

### Function: test_line_extent_compound_coords2(self)

### Function: test_line_extents_affine(self)

### Function: test_line_extents_non_affine(self)

### Function: test_pathc_extents_non_affine(self)

### Function: test_pathc_extents_affine(self)

### Function: test_line_extents_for_non_affine_transData(self)

### Function: __init__(self)

### Function: transform_path_non_affine(self, path)

### Function: transform_non_affine(self, path)

### Function: __init__(self, scale_factor)

### Function: _as_mpl_transform(self, axes)
