## AI Summary

A file named test_linalg.py.


### Function: consistent_subclass(out, in_)

### Function: assert_almost_equal(a, b, single_decimal, double_decimal)

### Function: get_real_dtype(dtype)

### Function: get_complex_dtype(dtype)

### Function: get_rtol(dtype)

## Class: LinalgCase

### Function: apply_tag(tag, cases)

**Description:** Add the given tag (a string) to each of the cases (a list of LinalgCase
objects)

### Function: _make_generalized_cases()

### Function: _stride_comb_iter(x)

**Description:** Generate cartesian product of strides for all axes

### Function: _make_strided_cases()

## Class: LinalgTestCase

## Class: LinalgSquareTestCase

## Class: LinalgNonsquareTestCase

## Class: HermitianTestCase

## Class: LinalgGeneralizedSquareTestCase

## Class: LinalgGeneralizedNonsquareTestCase

## Class: HermitianGeneralizedTestCase

### Function: identity_like_generalized(a)

## Class: SolveCases

## Class: TestSolve

## Class: InvCases

## Class: TestInv

## Class: EigvalsCases

## Class: TestEigvals

## Class: EigCases

## Class: TestEig

## Class: SVDBaseTests

## Class: SVDCases

## Class: TestSVD

## Class: SVDHermitianCases

## Class: TestSVDHermitian

## Class: CondCases

## Class: TestCond

## Class: PinvCases

## Class: TestPinv

## Class: PinvHermitianCases

## Class: TestPinvHermitian

### Function: test_pinv_rtol_arg()

## Class: DetCases

## Class: TestDet

## Class: LstsqCases

## Class: TestLstsq

## Class: TestMatrixPower

## Class: TestEigvalshCases

## Class: TestEigvalsh

## Class: TestEighCases

## Class: TestEigh

## Class: _TestNormBase

## Class: _TestNormGeneral

## Class: _TestNorm2D

## Class: _TestNorm

## Class: TestNorm_NonSystematic

## Class: _TestNormDoubleBase

## Class: _TestNormSingleBase

## Class: _TestNormInt64Base

## Class: TestNormDouble

## Class: TestNormSingle

## Class: TestNormInt64

## Class: TestMatrixRank

### Function: test_reduced_rank()

## Class: TestQR

## Class: TestCholesky

## Class: TestOuter

### Function: test_byteorder_check()

### Function: test_generalized_raise_multiloop()

### Function: test_xerbla_override()

### Function: test_sdot_bug_8577()

## Class: TestMultiDot

## Class: TestTensorinv

## Class: TestTensorsolve

### Function: test_unsupported_commontype()

### Function: test_blas64_dot()

### Function: test_blas64_geqrf_lwork_smoketest()

### Function: test_diagonal()

### Function: test_trace()

### Function: test_cross()

### Function: test_tensordot()

### Function: test_matmul()

### Function: test_matrix_transpose()

### Function: test_matrix_norm()

### Function: test_vector_norm()

### Function: __init__(self, name, a, b, tags)

**Description:** A bundle of arguments to be passed to a test case, with an identifying
name, the operands a and b, and a set of tags to filter the tests

### Function: check(self, do)

**Description:** Run the function `do` on this test case, expanding arguments

### Function: __repr__(self)

### Function: check_cases(self, require, exclude)

**Description:** Run func on each of the cases with all of the tags in require, and none
of the tags in exclude

### Function: test_sq_cases(self)

### Function: test_empty_sq_cases(self)

### Function: test_nonsq_cases(self)

### Function: test_empty_nonsq_cases(self)

### Function: test_herm_cases(self)

### Function: test_empty_herm_cases(self)

### Function: test_generalized_sq_cases(self)

### Function: test_generalized_empty_sq_cases(self)

### Function: test_generalized_nonsq_cases(self)

### Function: test_generalized_empty_nonsq_cases(self)

### Function: test_generalized_herm_cases(self)

### Function: test_generalized_empty_herm_cases(self)

### Function: do(self, a, b, tags)

### Function: test_types(self, dtype)

### Function: test_1_d(self)

### Function: test_0_size(self)

### Function: test_0_size_k(self)

### Function: do(self, a, b, tags)

### Function: test_types(self, dtype)

### Function: test_0_size(self)

### Function: do(self, a, b, tags)

### Function: test_types(self, dtype)

### Function: test_0_size(self)

### Function: do(self, a, b, tags)

### Function: test_types(self, dtype)

### Function: test_0_size(self)

### Function: test_types(self, dtype)

### Function: do(self, a, b, tags)

### Function: test_empty_identity(self)

**Description:** Empty input should put an identity matrix in u or vh 

### Function: test_svdvals(self)

### Function: do(self, a, b, tags)

### Function: do(self, a, b, tags)

### Function: test_basic_nonsvd(self)

### Function: test_singular(self)

### Function: test_nan(self)

### Function: test_stacked_singular(self)

### Function: do(self, a, b, tags)

### Function: do(self, a, b, tags)

### Function: do(self, a, b, tags)

### Function: test_zero(self)

### Function: test_types(self, dtype)

### Function: test_0_size(self)

### Function: do(self, a, b, tags)

### Function: test_rcond(self)

### Function: test_empty_a_b(self, m, n, n_rhs)

### Function: test_incompatible_dims(self)

### Function: test_large_power(self, dt)

### Function: test_power_is_zero(self, dt)

### Function: test_power_is_one(self, dt)

### Function: test_power_is_two(self, dt)

### Function: test_power_is_minus_one(self, dt)

### Function: test_exceptions_bad_power(self, dt)

### Function: test_exceptions_non_square(self, dt)

### Function: test_exceptions_not_invertible(self, dt)

### Function: do(self, a, b, tags)

### Function: test_types(self, dtype)

### Function: test_invalid(self)

### Function: test_UPLO(self)

### Function: test_0_size(self)

### Function: do(self, a, b, tags)

### Function: test_types(self, dtype)

### Function: test_invalid(self)

### Function: test_UPLO(self)

### Function: test_0_size(self)

### Function: check_dtype(x, res)

### Function: test_empty(self)

### Function: test_vector_return_type(self)

### Function: test_vector(self)

### Function: test_axis(self)

### Function: test_keepdims(self)

### Function: test_matrix_empty(self)

### Function: test_matrix_return_type(self)

### Function: test_matrix_2x2(self)

### Function: test_matrix_3x3(self)

### Function: test_bad_args(self)

### Function: test_longdouble_norm(self)

### Function: test_intmin(self)

### Function: test_complex_high_ord(self)

### Function: test_matrix_rank(self)

### Function: test_symmetric_rank(self)

### Function: check_qr(self, a)

### Function: test_qr_empty(self, m, n)

### Function: test_mode_raw(self)

### Function: test_mode_all_but_economic(self)

### Function: check_qr_stacked(self, a)

### Function: test_stacked_inputs(self, outer_size, size, dt)

### Function: test_basic_property(self, shape, dtype, upper)

### Function: test_0_size(self)

### Function: test_upper_lower_arg(self)

### Function: test_basic_function_with_three_arguments(self)

### Function: test_basic_function_with_two_arguments(self)

### Function: test_basic_function_with_dynamic_programming_optimization(self)

### Function: test_vector_as_first_argument(self)

### Function: test_vector_as_last_argument(self)

### Function: test_vector_as_first_and_last_argument(self)

### Function: test_three_arguments_and_out(self)

### Function: test_two_arguments_and_out(self)

### Function: test_dynamic_programming_optimization_and_out(self)

### Function: test_dynamic_programming_logic(self)

### Function: test_too_few_input_arrays(self)

### Function: test_non_square_handling(self, arr, ind)

### Function: test_tensorinv_shape(self, shape, ind)

### Function: test_tensorinv_ind_limit(self, ind)

### Function: test_tensorinv_result(self)

### Function: test_non_square_handling(self, a, axes)

### Function: test_tensorsolve_result(self, shape)

## Class: ArraySubclass

## Class: ArraySubclass

## Class: ArraySubclass

## Class: ArraySubclass

## Class: ArraySubclass

## Class: ArraySubclass

### Function: hermitian(mat)

### Function: tz(M)

### Function: tz(mat)

### Function: tz(mat)

### Function: tz(mat)

## Class: ArraySubclass

## Class: ArraySubclass

### Function: _test(v)

## Class: ArraySubclass
