## AI Summary

A file named test_nanfunctions.py.


## Class: TestSignatureMatch

## Class: TestNanFunctions_MinMax

## Class: TestNanFunctions_ArgminArgmax

## Class: TestNanFunctions_NumberTypes

## Class: SharedNanFunctionsTestsMixin

## Class: TestNanFunctions_SumProd

## Class: TestNanFunctions_CumSumProd

## Class: TestNanFunctions_MeanVarStd

## Class: TestNanFunctions_Median

## Class: TestNanFunctions_Percentile

## Class: TestNanFunctions_Quantile

### Function: test__nan_mask(arr, expected)

### Function: test__replace_nan()

**Description:** Test that _replace_nan returns the original array if there are no
NaNs, not a copy.

### Function: get_signature(func, default)

**Description:** Construct a signature and replace all default parameter-values.

### Function: test_signature_match(self, nan_func, func)

### Function: test_exhaustiveness(self)

**Description:** Validate that all nan functions are actually tested.

### Function: test_mutation(self)

### Function: test_keepdims(self)

### Function: test_out(self)

### Function: test_dtype_from_input(self)

### Function: test_result_values(self)

### Function: test_allnans(self, axis, dtype, array)

### Function: test_masked(self)

### Function: test_scalar(self)

### Function: test_subclass(self)

### Function: test_object_array(self)

### Function: test_initial(self, dtype)

### Function: test_where(self, dtype)

### Function: test_mutation(self)

### Function: test_result_values(self)

### Function: test_allnans(self, axis, dtype, array)

### Function: test_empty(self)

### Function: test_scalar(self)

### Function: test_subclass(self)

### Function: test_keepdims(self, dtype)

### Function: test_out(self, dtype)

### Function: test_nanfunc(self, mat, dtype, nanfunc, func)

### Function: test_nanfunc_q(self, mat, dtype, nanfunc, func)

### Function: test_nanfunc_ddof(self, mat, dtype, nanfunc, func)

### Function: test_nanfunc_correction(self, mat, dtype, nanfunc)

### Function: test_mutation(self)

### Function: test_keepdims(self)

### Function: test_out(self)

### Function: test_dtype_from_dtype(self)

### Function: test_dtype_from_char(self)

### Function: test_dtype_from_input(self)

### Function: test_result_values(self)

### Function: test_scalar(self)

### Function: test_subclass(self)

### Function: test_allnans(self, axis, dtype, array)

### Function: test_empty(self)

### Function: test_initial(self, dtype)

### Function: test_where(self, dtype)

### Function: test_allnans(self, axis, dtype, array)

### Function: test_empty(self)

### Function: test_keepdims(self)

### Function: test_result_values(self)

### Function: test_out(self)

### Function: test_dtype_error(self)

### Function: test_out_dtype_error(self)

### Function: test_ddof(self)

### Function: test_ddof_too_big(self)

### Function: test_allnans(self, axis, dtype, array)

### Function: test_empty(self)

### Function: test_where(self, dtype)

### Function: test_nanstd_with_mean_keyword(self)

### Function: test_mutation(self)

### Function: test_keepdims(self)

### Function: test_keepdims_out(self, axis)

### Function: test_out(self)

### Function: test_small_large(self)

### Function: test_result_values(self)

### Function: test_allnans(self, dtype, axis)

### Function: test_empty(self)

### Function: test_scalar(self)

### Function: test_extended_axis_invalid(self)

### Function: test_float_special(self)

### Function: test_mutation(self)

### Function: test_keepdims(self)

### Function: test_keepdims_out(self, q, axis)

### Function: test_out(self, weighted)

### Function: test_complex(self)

### Function: test_result_values(self, weighted, use_out)

### Function: test_allnans(self, axis, dtype, array)

### Function: test_empty(self)

### Function: test_scalar(self)

### Function: test_extended_axis_invalid(self)

### Function: test_multiple_percentiles(self)

### Function: test_nan_value_with_weight(self, nan_weight)

### Function: test_nan_value_with_weight_ndim(self, axis)

### Function: test_regression(self, weighted)

### Function: test_basic(self)

### Function: test_complex(self)

### Function: test_no_p_overwrite(self)

### Function: test_allnans(self, axis, dtype, array)

## Class: MyNDArray

## Class: MyNDArray

## Class: MyNDArray

## Class: MyNDArray

## Class: MyNDArray

### Function: gen_weights(d)

### Function: gen_weights(d)
