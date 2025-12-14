## AI Summary

A file named test_mlab.py.


### Function: test_window()

## Class: TestDetrend

## Class: TestSpectral

### Function: test_cohere()

## Class: TestGaussianKDE

## Class: TestGaussianKDECustom

## Class: TestGaussianKDEEvaluate

### Function: test_psd_onesided_norm()

### Function: test_psd_oversampling()

**Description:** Test the case len(x) < NFFT for psd().

### Function: setup_method(self)

### Function: allclose(self)

### Function: test_detrend_none(self)

### Function: test_detrend_mean(self)

### Function: test_detrend_mean_1d_base_slope_off_list_andor_axis0(self)

### Function: test_detrend_mean_2d(self)

### Function: test_detrend_ValueError(self)

### Function: test_detrend_mean_ValueError(self)

### Function: test_detrend_linear(self)

### Function: test_detrend_str_linear_1d(self)

### Function: test_detrend_linear_2d(self)

### Function: stim(self, request, fstims, iscomplex, sides, len_x, NFFT_density, nover_density, pad_to_density, pad_to_spectrum)

### Function: check_freqs(self, vals, targfreqs, resfreqs, fstims)

### Function: check_maxfreq(self, spec, fsp, fstims)

### Function: test_spectral_helper_raises(self)

### Function: test_single_spectrum_helper_unsupported_modes(self, mode)

### Function: test_spectral_helper_psd(self, mode, case)

### Function: test_csd(self)

### Function: test_csd_padding(self)

**Description:** Test zero padding of csd().

### Function: test_psd(self)

### Function: test_psd_detrend(self, make_data, detrend)

### Function: test_psd_window_hanning(self)

### Function: test_psd_window_hanning_detrend_linear(self)

### Function: test_psd_window_flattop(self)

### Function: test_psd_windowarray(self)

### Function: test_psd_windowarray_scale_by_freq(self)

### Function: test_spectrum(self, kind)

### Function: test_specgram(self, kwargs)

### Function: test_specgram_warn_only1seg(self)

**Description:** Warning should be raised if len(x) <= NFFT.

### Function: test_psd_csd_equal(self)

### Function: test_specgram_auto_default_psd_equal(self, mode)

**Description:** Test that mlab.specgram without mode and with mode 'default' and 'psd'
are all the same.

### Function: test_specgram_complex_equivalent(self, mode, conv)

### Function: test_psd_windowarray_equal(self)

### Function: test_kde_integer_input(self)

**Description:** Regression test for #1181.

### Function: test_gaussian_kde_covariance_caching(self)

### Function: test_kde_bandwidth_method(self)

### Function: test_no_data(self)

**Description:** Pass no data into the GaussianKDE class.

### Function: test_single_dataset_element(self)

**Description:** Pass a single dataset element into the GaussianKDE class.

### Function: test_silverman_multidim_dataset(self)

**Description:** Test silverman's for a multi-dimensional array.

### Function: test_silverman_singledim_dataset(self)

**Description:** Test silverman's output for a single dimension list.

### Function: test_scott_multidim_dataset(self)

**Description:** Test scott's output for a multi-dimensional array.

### Function: test_scott_singledim_dataset(self)

**Description:** Test scott's output a single-dimensional array.

### Function: test_scalar_empty_dataset(self)

**Description:** Test the scalar's cov factor for an empty array.

### Function: test_scalar_covariance_dataset(self)

**Description:** Test a scalar's cov factor.

### Function: test_callable_covariance_dataset(self)

**Description:** Test the callable's cov factor for a multi-dimensional array.

### Function: test_callable_singledim_dataset(self)

**Description:** Test the callable's cov factor for a single-dimensional array.

### Function: test_wrong_bw_method(self)

**Description:** Test the error message that should be called when bw is invalid.

### Function: test_evaluate_diff_dim(self)

**Description:** Test the evaluate method when the dim's of dataset and points have
different dimensions.

### Function: test_evaluate_inv_dim(self)

**Description:** Invert the dimensions; i.e., for a dataset of dimension 1 [3, 2, 4],
the points should have a dimension of 3 [[3], [2], [4]].

### Function: test_evaluate_dim_and_num(self)

**Description:** Tests if evaluated against a one by one array

### Function: test_evaluate_point_dim_not_one(self)

### Function: test_evaluate_equal_dim_and_num_lt(self)

### Function: callable_fun(x)
