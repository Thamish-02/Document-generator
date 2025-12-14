## AI Summary

A file named test_scalarmath.py.


## Class: TestTypes

### Function: check_ufunc_scalar_equivalence(op, arr1, arr2)

### Function: test_array_scalar_ufunc_equivalence(op, arr1, arr2)

**Description:** This is a thorough test attempting to cover important promotion paths
and ensuring that arrays and scalars stay as aligned as possible.
However, if it creates troubles, it should maybe just be removed.

### Function: test_array_scalar_ufunc_dtypes(op, dt1, dt2)

### Function: test_int_float_promotion_truediv(fscalar)

## Class: TestBaseMath

## Class: TestPower

### Function: floordiv_and_mod(x, y)

### Function: _signs(dt)

## Class: TestModulus

## Class: TestComplexDivision

## Class: TestConversion

## Class: TestRepr

## Class: TestMultiply

## Class: TestNegative

## Class: TestSubtract

## Class: TestAbs

## Class: TestBitShifts

## Class: TestHash

### Function: recursionlimit(n)

### Function: test_operator_object_left(o, op, type_)

### Function: test_operator_object_right(o, op, type_)

### Function: test_operator_scalars(op, type1, type2)

### Function: test_longdouble_operators_with_obj(sctype, op)

### Function: test_longdouble_with_arrlike(sctype, op)

### Function: test_longdouble_operators_with_large_int(sctype, op)

### Function: test_scalar_integer_operation_overflow(dtype, operation)

### Function: test_scalar_signed_integer_overflow(dtype, operation)

### Function: test_scalar_unsigned_integer_overflow(dtype)

### Function: test_scalar_integer_operation_divbyzero(dtype, operation)

### Function: test_subclass_deferral(sctype, __op__, __rop__, op, cmp)

**Description:** This test covers scalar subclass deferral.  Note that this is exceedingly
complicated, especially since it tends to fall back to the array paths and
these additionally add the "array priority" mechanism.

The behaviour was modified subtly in 1.22 (to make it closer to how Python
scalars work).  Due to its complexity and the fact that subclassing NumPy
scalars is probably a bad idea to begin with.  There is probably room
for adjustments here.

### Function: test_longdouble_complex()

### Function: test_pyscalar_subclasses(subtype, __op__, __rop__, op, cmp)

### Function: test_truediv_int()

### Function: test_scalar_matches_array_op_with_pyscalar(op, sctype, other_type, rop)

### Function: test_types(self)

### Function: test_type_add(self)

### Function: test_type_create(self)

### Function: test_leak(self)

### Function: test_blocked(self)

### Function: test_lower_align(self)

### Function: test_small_types(self)

### Function: test_large_types(self)

### Function: test_integers_to_negative_integer_power(self)

### Function: test_mixed_types(self)

### Function: test_modular_power(self)

### Function: test_modulus_basic(self)

### Function: test_float_modulus_exact(self)

### Function: test_float_modulus_roundoff(self)

### Function: test_float_modulus_corner_cases(self)

### Function: test_inplace_floordiv_handling(self)

### Function: test_zero_division(self)

### Function: test_signed_zeros(self)

### Function: test_branches(self)

### Function: test_int_from_long(self)

### Function: test_iinfo_long_values(self)

### Function: test_int_raise_behaviour(self)

### Function: test_int_from_infinite_longdouble(self)

### Function: test_int_from_infinite_longdouble___int__(self)

### Function: test_int_from_huge_longdouble(self)

### Function: test_int_from_longdouble(self)

### Function: test_numpy_scalar_relational_operators(self)

### Function: test_scalar_comparison_to_none(self)

### Function: _test_type_repr(self, t)

### Function: test_float_repr(self)

## Class: TestSizeOf

### Function: test_seq_repeat(self)

### Function: test_no_seq_repeat_basic_array_like(self)

### Function: test_exceptions(self)

### Function: test_result(self)

### Function: test_exceptions(self)

### Function: test_result(self)

### Function: _test_abs_func(self, absfunc, test_dtype)

### Function: test_builtin_abs(self, dtype)

### Function: test_numpy_abs(self, dtype)

### Function: test_shift_all_bits(self, type_code, op)

**Description:** Shifts where the shift amount is the width of the type or wider 

### Function: test_integer_hashes(self, type_code)

### Function: test_float_and_complex_hashes(self, type_code)

### Function: test_complex_hashes(self, type_code)

## Class: myf_simple1

## Class: myf_simple2

### Function: op_func(self, other)

### Function: rop_func(self, other)

### Function: op_func(self, other)

### Function: rop_func(self, other)

### Function: overflow_error_func(dtype)

### Function: test_equal_nbytes(self)

### Function: test_error(self)

## Class: ArrayLike

### Function: __init__(self, arr)

### Function: __array__(self, dtype, copy)
