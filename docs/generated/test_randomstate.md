## AI Summary

A file named test_randomstate.py.


### Function: int_func(request)

### Function: restore_singleton_bitgen()

**Description:** Ensures that the singleton bitgen is restored after a test

### Function: assert_mt19937_state_equal(a, b)

## Class: TestSeed

## Class: TestBinomial

## Class: TestMultinomial

## Class: TestSetState

## Class: TestRandint

## Class: TestRandomDist

## Class: TestBroadcast

## Class: TestThread

## Class: TestSingleEltArrayInput

### Function: test_integer_dtype(int_func)

### Function: test_integer_repeat(int_func)

### Function: test_broadcast_size_error()

### Function: test_randomstate_ctor_old_style_pickle()

### Function: test_hot_swap(restore_singleton_bitgen)

### Function: test_seed_alt_bit_gen(restore_singleton_bitgen)

### Function: test_state_error_alt_bit_gen(restore_singleton_bitgen)

### Function: test_swap_worked(restore_singleton_bitgen)

### Function: test_swapped_singleton_against_direct(restore_singleton_bitgen)

### Function: test_scalar(self)

### Function: test_array(self)

### Function: test_invalid_scalar(self)

### Function: test_invalid_array(self)

### Function: test_invalid_array_shape(self)

### Function: test_cannot_seed(self)

### Function: test_invalid_initialization(self)

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

### Function: test_multinomial_n_float(self)

### Function: setup_method(self)

### Function: test_basic(self)

### Function: test_gaussian_reset(self)

### Function: test_gaussian_reset_in_media_res(self)

### Function: test_backwards_compatibility(self)

### Function: test_negative_binomial(self)

### Function: test_get_state_warning(self)

### Function: test_invalid_legacy_state_setting(self)

### Function: test_pickle(self)

### Function: test_state_setting(self)

### Function: test_repr(self)

### Function: test_unsupported_type(self)

### Function: test_bounds_checking(self)

### Function: test_rng_zero_and_extremes(self)

### Function: test_full_range(self)

### Function: test_in_bounds_fuzz(self)

### Function: test_repeatability(self)

### Function: test_repeatability_32bit_boundary_broadcasting(self)

### Function: test_int64_uint64_corner_case(self)

### Function: test_respect_dtype_singleton(self)

### Function: setup_method(self)

### Function: test_rand(self)

### Function: test_rand_singleton(self)

### Function: test_randn(self)

### Function: test_randint(self)

### Function: test_random_integers(self)

### Function: test_tomaxint(self)

### Function: test_random_integers_max_int(self)

### Function: test_random_integers_deprecated(self)

### Function: test_random_sample(self)

### Function: test_choice_uniform_replace(self)

### Function: test_choice_nonuniform_replace(self)

### Function: test_choice_uniform_noreplace(self)

### Function: test_choice_nonuniform_noreplace(self)

### Function: test_choice_noninteger(self)

### Function: test_choice_exceptions(self)

### Function: test_choice_return_shape(self)

### Function: test_choice_nan_probabilities(self)

### Function: test_choice_p_non_contiguous(self)

### Function: test_bytes(self)

### Function: test_shuffle(self)

### Function: test_shuffle_masked(self)

### Function: test_permutation(self)

### Function: test_beta(self)

### Function: test_binomial(self)

### Function: test_chisquare(self)

### Function: test_dirichlet(self)

### Function: test_dirichlet_size(self)

### Function: test_dirichlet_bad_alpha(self)

### Function: test_dirichlet_alpha_non_contiguous(self)

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

### Function: test_multivariate_normal(self)

### Function: test_negative_binomial(self)

### Function: test_negative_binomial_exceptions(self)

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

### Function: test_standard_gamma(self)

### Function: test_standard_gamma_0(self)

### Function: test_standard_normal(self)

### Function: test_randn_singleton(self)

### Function: test_standard_t(self)

### Function: test_triangular(self)

### Function: test_uniform(self)

### Function: test_uniform_range_bounds(self)

### Function: test_scalar_exception_propagation(self)

### Function: test_vonmises(self)

### Function: test_vonmises_small(self)

### Function: test_vonmises_large(self)

### Function: test_vonmises_nan(self)

### Function: test_wald(self)

### Function: test_weibull(self)

### Function: test_weibull_0(self)

### Function: test_zipf(self)

### Function: setup_method(self)

### Function: set_seed(self)

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

### Function: setup_method(self)

### Function: check_function(self, function, sz)

### Function: test_normal(self)

### Function: test_exp(self)

### Function: test_multinomial(self)

### Function: setup_method(self)

### Function: test_one_arg_funcs(self)

### Function: test_two_arg_funcs(self)

### Function: test_three_arg_funcs(self)

### Function: test_shuffle_invalid_objects(self)

## Class: ThrowingFloat

## Class: ThrowingInteger

### Function: gen_random(state, out)

### Function: gen_random(state, out)

### Function: gen_random(state, out)

### Function: __float__(self)

### Function: __int__(self)
