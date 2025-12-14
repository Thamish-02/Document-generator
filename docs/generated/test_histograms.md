## AI Summary

A file named test_histograms.py.


## Class: TestHistogram

## Class: TestHistogramOptimBinNums

**Description:** Provide test coverage when using provided estimators for optimal number of
bins

## Class: TestHistogramdd

### Function: setup_method(self)

### Function: teardown_method(self)

### Function: test_simple(self)

### Function: test_one_bin(self)

### Function: test_density(self)

### Function: test_outliers(self)

### Function: test_arr_weights_mismatch(self)

### Function: test_type(self)

### Function: test_f32_rounding(self)

### Function: test_bool_conversion(self)

### Function: test_weights(self)

### Function: test_exotic_weights(self)

### Function: test_no_side_effects(self)

### Function: test_empty(self)

### Function: test_error_binnum_type(self)

### Function: test_finite_range(self)

### Function: test_invalid_range(self)

### Function: test_bin_edge_cases(self)

### Function: test_last_bin_inclusive_range(self)

### Function: test_bin_array_dims(self)

### Function: test_unsigned_monotonicity_check(self)

### Function: test_object_array_of_0d(self)

### Function: test_some_nan_values(self)

### Function: test_datetime(self)

### Function: do_signed_overflow_bounds(self, dtype)

### Function: test_signed_overflow_bounds(self)

### Function: do_precision_lower_bound(self, float_small, float_large)

### Function: do_precision_upper_bound(self, float_small, float_large)

### Function: do_precision(self, float_small, float_large)

### Function: test_precision(self)

### Function: test_histogram_bin_edges(self)

### Function: test_small_value_range(self)

### Function: test_big_arrays(self)

### Function: test_gh_23110(self)

### Function: test_empty(self)

### Function: test_simple(self)

**Description:** Straightforward testing with a mixture of linspace data (for
consistency). All test values have been precomputed and the values
shouldn't change

### Function: test_small(self)

**Description:** Smaller datasets have the potential to cause issues with the data
adaptive methods, especially the FD method. All bin numbers have been
precalculated.

### Function: test_incorrect_methods(self)

**Description:** Check a Value Error is thrown when an unknown string is passed in

### Function: test_novariance(self)

**Description:** Check that methods handle no variance in data
Primarily for Scott and FD as the SD and IQR are both 0 in this case

### Function: test_limited_variance(self)

**Description:** Check when IQR is 0, but variance exists, we return the sturges value
and not the fd value.

### Function: test_outlier(self)

**Description:** Check the FD, Scott and Doane with outliers.

The FD estimates a smaller binwidth since it's less affected by
outliers. Since the range is so (artificially) large, this means more
bins, most of which will be empty, but the data of interest usually is
unaffected. The Scott estimator is more affected and returns fewer bins,
despite most of the variance being in one area of the data. The Doane
estimator lies somewhere between the other two.

### Function: test_scott_vs_stone(self)

**Description:** Verify that Scott's rule and Stone's rule converges for normally distributed data

### Function: test_simple_range(self)

**Description:** Straightforward testing with a mixture of linspace data (for
consistency). Adding in a 3rd mixture that will then be
completely ignored. All test values have been precomputed and
the shouldn't change.

### Function: test_signed_integer_data(self, bins)

### Function: test_integer(self, bins)

**Description:** Test that bin width for integer data is at least 1.

### Function: test_integer_non_auto(self)

**Description:** Test that the bin-width>=1 requirement *only* applies to auto binning.

### Function: test_simple_weighted(self)

**Description:** Check that weighted data raises a TypeError

### Function: test_simple(self)

### Function: test_shape_3d(self)

### Function: test_shape_4d(self)

### Function: test_weights(self)

### Function: test_identical_samples(self)

### Function: test_empty(self)

### Function: test_bins_errors(self)

### Function: test_inf_edges(self)

### Function: test_rightmost_binedge(self)

### Function: test_finite_range(self)

### Function: test_equal_edges(self)

**Description:** Test that adjacent entries in an edge array can be equal 

### Function: test_edge_dtype(self)

**Description:** Test that if an edge array is input, its type is preserved 

### Function: test_large_integers(self)

### Function: test_density_non_uniform_2d(self)

### Function: test_density_non_uniform_1d(self)

### Function: nbins_ratio(seed, size)
