## AI Summary

A file named test_generator_mt19937.py.


### Function: endpoint(request)

## Class: TestSeed

## Class: TestBinomial

## Class: TestMultinomial

## Class: TestMultivariateHypergeometric

## Class: TestSetState

## Class: TestIntegers

## Class: TestRandomDist

## Class: TestBroadcast

## Class: TestThread

## Class: TestSingleEltArrayInput

### Function: test_jumped(config)

### Function: test_broadcast_size_error()

### Function: test_broadcast_size_scalar()

### Function: test_ragged_shuffle()

### Function: test_single_arg_integer_exception(high, endpoint)

### Function: test_c_contig_req_out(dtype)

### Function: test_contig_req_out(dist, order, dtype)

### Function: test_generator_ctor_old_style_pickle()

### Function: test_pickle_preserves_seed_sequence()

### Function: test_legacy_pickle(version)

### Function: test_scalar(self)

### Function: test_array(self)

### Function: test_seedsequence(self)

### Function: test_invalid_scalar(self)

### Function: test_invalid_array(self)

### Function: test_noninstantized_bitgen(self)

### Function: test_n_zero(self)

### Function: test_p_is_nan(self)

### Function: test_basic(self)

### Function: test_zero_probability(self)

### Function: test_int_negative_interval(self)

### Function: test_size(self)

### Function: test_invalid_prob(self)

### Function: test_invalid_n(self)

### Function: test_p_non_contiguous(self)

### Function: test_multinomial_pvals_float32(self)

### Function: setup_method(self)

### Function: test_argument_validation(self)

### Function: test_edge_cases(self, method)

### Function: test_typical_cases(self, nsample, method, size)

### Function: test_repeatability1(self)

### Function: test_repeatability2(self)

### Function: test_repeatability3(self)

### Function: setup_method(self)

### Function: test_gaussian_reset(self)

### Function: test_gaussian_reset_in_media_res(self)

### Function: test_negative_binomial(self)

### Function: test_unsupported_type(self, endpoint)

### Function: test_bounds_checking(self, endpoint)

### Function: test_bounds_checking_array(self, endpoint)

### Function: test_rng_zero_and_extremes(self, endpoint)

### Function: test_rng_zero_and_extremes_array(self, endpoint)

### Function: test_full_range(self, endpoint)

### Function: test_full_range_array(self, endpoint)

### Function: test_in_bounds_fuzz(self, endpoint)

### Function: test_scalar_array_equiv(self, endpoint)

### Function: test_repeatability(self, endpoint)

### Function: test_repeatability_broadcasting(self, endpoint)

### Function: test_repeatability_32bit_boundary(self, bound, expected)

### Function: test_repeatability_32bit_boundary_broadcasting(self)

### Function: test_int64_uint64_broadcast_exceptions(self, endpoint)

### Function: test_int64_uint64_corner_case(self, endpoint)

### Function: test_respect_dtype_singleton(self, endpoint)

### Function: test_respect_dtype_array(self, endpoint)

### Function: test_zero_size(self, endpoint)

### Function: test_error_byteorder(self)

### Function: test_integers_small_dtype_chisquared(self, sample_size, high, dtype, chi2max)

### Function: setup_method(self)

### Function: test_integers(self)

### Function: test_integers_masked(self)

### Function: test_integers_closed(self)

### Function: test_integers_max_int(self)

### Function: test_random(self)

### Function: test_random_float(self)

### Function: test_random_float_scalar(self)

### Function: test_random_distribution_of_lsb(self, dtype, uint_view_type)

### Function: test_random_unsupported_type(self)

### Function: test_choice_uniform_replace(self)

### Function: test_choice_nonuniform_replace(self)

### Function: test_choice_uniform_noreplace(self)

### Function: test_choice_nonuniform_noreplace(self)

### Function: test_choice_noninteger(self)

### Function: test_choice_multidimensional_default_axis(self)

### Function: test_choice_multidimensional_custom_axis(self)

### Function: test_choice_exceptions(self)

### Function: test_choice_return_shape(self)

### Function: test_choice_nan_probabilities(self)

### Function: test_choice_p_non_contiguous(self)

### Function: test_choice_return_type(self)

### Function: test_choice_large_sample(self)

### Function: test_choice_array_size_empty_tuple(self)

### Function: test_bytes(self)

### Function: test_shuffle(self)

### Function: test_shuffle_custom_axis(self)

### Function: test_shuffle_custom_axis_empty(self)

### Function: test_shuffle_axis_nonsquare(self)

### Function: test_shuffle_masked(self)

### Function: test_shuffle_exceptions(self)

### Function: test_shuffle_not_writeable(self)

### Function: test_permutation(self)

### Function: test_permutation_custom_axis(self)

### Function: test_permutation_exceptions(self)

### Function: test_permuted(self, dtype, axis, expected)

### Function: test_permuted_with_strides(self)

### Function: test_permuted_empty(self)

### Function: test_permuted_out_with_wrong_shape(self, outshape)

### Function: test_permuted_out_with_wrong_type(self)

### Function: test_permuted_not_writeable(self)

### Function: test_beta(self)

### Function: test_binomial(self)

### Function: test_chisquare(self)

### Function: test_dirichlet(self)

### Function: test_dirichlet_size(self)

### Function: test_dirichlet_bad_alpha(self)

### Function: test_dirichlet_alpha_non_contiguous(self)

### Function: test_dirichlet_small_alpha(self)

### Function: test_dirichlet_moderately_small_alpha(self)

### Function: test_dirichlet_multiple_zeros_in_alpha(self, alpha)

### Function: test_exponential(self)

### Function: test_exponential_0(self)

### Function: test_f(self)

### Function: test_gamma(self)

### Function: test_gamma_0(self)

### Function: test_geometric(self)

### Function: test_geometric_exceptions(self)

### Function: test_gumbel(self)

### Function: test_gumbel_0(self)

### Function: test_hypergeometric(self)

### Function: test_laplace(self)

### Function: test_laplace_0(self)

### Function: test_logistic(self)

### Function: test_lognormal(self)

### Function: test_lognormal_0(self)

### Function: test_logseries(self)

### Function: test_logseries_zero(self)

### Function: test_logseries_exceptions(self, value)

### Function: test_multinomial(self)

### Function: test_multivariate_normal(self, method)

### Function: test_multivariate_normal_disallow_complex(self, mean, cov)

### Function: test_multivariate_normal_basic_stats(self, method)

### Function: test_negative_binomial(self)

### Function: test_negative_binomial_exceptions(self)

### Function: test_negative_binomial_p0_exception(self)

### Function: test_negative_binomial_invalid_p_n_combination(self)

### Function: test_noncentral_chisquare(self)

### Function: test_noncentral_f(self)

### Function: test_noncentral_f_nan(self)

### Function: test_normal(self)

### Function: test_normal_0(self)

### Function: test_pareto(self)

### Function: test_poisson(self)

### Function: test_poisson_exceptions(self)

### Function: test_power(self)

### Function: test_rayleigh(self)

### Function: test_rayleigh_0(self)

### Function: test_standard_cauchy(self)

### Function: test_standard_exponential(self)

### Function: test_standard_expoential_type_error(self)

### Function: test_standard_gamma(self)

### Function: test_standard_gammma_scalar_float(self)

### Function: test_standard_gamma_float(self)

### Function: test_standard_gammma_float_out(self)

### Function: test_standard_gamma_unknown_type(self)

### Function: test_out_size_mismatch(self)

### Function: test_standard_gamma_0(self)

### Function: test_standard_normal(self)

### Function: test_standard_normal_unsupported_type(self)

### Function: test_standard_t(self)

### Function: test_triangular(self)

### Function: test_uniform(self)

### Function: test_uniform_range_bounds(self)

### Function: test_uniform_zero_range(self)

### Function: test_uniform_neg_range(self)

### Function: test_scalar_exception_propagation(self)

### Function: test_vonmises(self)

### Function: test_vonmises_small(self)

### Function: test_vonmises_nan(self)

### Function: test_vonmises_large_kappa(self, kappa)

### Function: test_vonmises_large_kappa_range(self, mu, kappa)

### Function: test_wald(self)

### Function: test_weibull(self)

### Function: test_weibull_0(self)

### Function: test_zipf(self)

### Function: setup_method(self)

### Function: test_uniform(self)

### Function: test_normal(self)

### Function: test_beta(self)

### Function: test_exponential(self)

### Function: test_standard_gamma(self)

### Function: test_gamma(self)

### Function: test_f(self)

### Function: test_noncentral_f(self)

### Function: test_noncentral_f_small_df(self)

### Function: test_chisquare(self)

### Function: test_noncentral_chisquare(self)

### Function: test_standard_t(self)

### Function: test_vonmises(self)

### Function: test_pareto(self)

### Function: test_weibull(self)

### Function: test_power(self)

### Function: test_laplace(self)

### Function: test_gumbel(self)

### Function: test_logistic(self)

### Function: test_lognormal(self)

### Function: test_rayleigh(self)

### Function: test_wald(self)

### Function: test_triangular(self)

### Function: test_binomial(self)

### Function: test_negative_binomial(self)

### Function: test_poisson(self)

### Function: test_zipf(self)

### Function: test_geometric(self)

### Function: test_hypergeometric(self)

### Function: test_logseries(self)

### Function: test_multinomial(self)

### Function: test_multinomial_pval_broadcast(self, n)

### Function: test_invalid_pvals_broadcast(self)

### Function: test_empty_outputs(self)

### Function: setup_method(self)

### Function: check_function(self, function, sz)

### Function: test_normal(self)

### Function: test_exp(self)

### Function: test_multinomial(self)

### Function: setup_method(self)

### Function: test_one_arg_funcs(self)

### Function: test_two_arg_funcs(self)

### Function: test_integers(self, endpoint)

### Function: test_three_arg_funcs(self)

## Class: ThrowingFloat

## Class: ThrowingInteger

### Function: gen_random(state, out)

### Function: gen_random(state, out)

### Function: gen_random(state, out)

### Function: __float__(self)

### Function: __int__(self)
