## AI Summary

A file named test_numerictypes.py.


### Function: normalize_descr(descr)

**Description:** Normalize a description adding the platform byteorder.

## Class: CreateZeros

**Description:** Check the creation of heterogeneous arrays zero-valued

## Class: TestCreateZerosPlain

**Description:** Check the creation of heterogeneous arrays zero-valued (plain)

## Class: TestCreateZerosNested

**Description:** Check the creation of heterogeneous arrays zero-valued (nested)

## Class: CreateValues

**Description:** Check the creation of heterogeneous arrays with values

## Class: TestCreateValuesPlainSingle

**Description:** Check the creation of heterogeneous arrays (plain, single row)

## Class: TestCreateValuesPlainMultiple

**Description:** Check the creation of heterogeneous arrays (plain, multiple rows)

## Class: TestCreateValuesNestedSingle

**Description:** Check the creation of heterogeneous arrays (nested, single row)

## Class: TestCreateValuesNestedMultiple

**Description:** Check the creation of heterogeneous arrays (nested, multiple rows)

## Class: ReadValuesPlain

**Description:** Check the reading of values in heterogeneous arrays (plain)

## Class: TestReadValuesPlainSingle

**Description:** Check the creation of heterogeneous arrays (plain, single row)

## Class: TestReadValuesPlainMultiple

**Description:** Check the values of heterogeneous arrays (plain, multiple rows)

## Class: ReadValuesNested

**Description:** Check the reading of values in heterogeneous arrays (nested)

## Class: TestReadValuesNestedSingle

**Description:** Check the values of heterogeneous arrays (nested, single row)

## Class: TestReadValuesNestedMultiple

**Description:** Check the values of heterogeneous arrays (nested, multiple rows)

## Class: TestEmptyField

## Class: TestMultipleFields

## Class: TestIsSubDType

## Class: TestIsDType

**Description:** Check correctness of `np.isdtype`. The test considers different argument
configurations: `np.isdtype(dtype, k1)` and `np.isdtype(dtype, (k1, k2))`
with concrete dtypes and dtype groups.

## Class: TestSctypeDict

## Class: TestMaximumSctype

## Class: Test_sctype2char

### Function: test_issctype(rep, expected)

## Class: TestDocStrings

## Class: TestScalarTypeNames

## Class: TestBoolDefinition

### Function: test_zeros0D(self)

**Description:** Check creation of 0-dimensional objects

### Function: test_zerosSD(self)

**Description:** Check creation of single-dimensional objects

### Function: test_zerosMD(self)

**Description:** Check creation of multi-dimensional objects

### Function: test_tuple(self)

**Description:** Check creation from tuples

### Function: test_list_of_tuple(self)

**Description:** Check creation from list of tuples

### Function: test_list_of_list_of_tuple(self)

**Description:** Check creation from list of list of tuples

### Function: test_access_fields(self)

### Function: test_access_top_fields(self)

**Description:** Check reading the top fields of a nested array

### Function: test_nested1_acessors(self)

**Description:** Check reading the nested fields of a nested array (1st level)

### Function: test_nested2_acessors(self)

**Description:** Check reading the nested fields of a nested array (2nd level)

### Function: test_nested1_descriptor(self)

**Description:** Check access nested descriptors of a nested array (1st level)

### Function: test_nested2_descriptor(self)

**Description:** Check access nested descriptors of a nested array (2nd level)

### Function: test_assign(self)

### Function: setup_method(self)

### Function: _bad_call(self)

### Function: test_no_tuple(self)

### Function: test_return(self)

### Function: test_both_abstract(self)

### Function: test_same(self)

### Function: test_subclass(self)

### Function: test_subclass_backwards(self)

### Function: test_sibling_class(self)

### Function: test_nondtype_nonscalartype(self)

### Function: test_isdtype(self, dtype, close_dtype, dtype_group)

### Function: test_isdtype_invalid_args(self)

### Function: test_sctypes_complete(self)

### Function: test_longdouble(self)

### Function: test_ulong(self)

### Function: test_int(self, t)

### Function: test_uint(self, t)

### Function: test_float(self, t)

### Function: test_complex(self, t)

### Function: test_other(self, t)

### Function: test_scalar_type(self)

### Function: test_other_type(self)

### Function: test_third_party_scalar_type(self)

### Function: test_array_instance(self)

### Function: test_abstract_type(self)

### Function: test_non_type(self)

### Function: test_platform_dependent_aliases(self)

### Function: test_names_are_unique(self)

### Function: test_names_reflect_attributes(self, t)

**Description:** Test that names correspond to where the type is under ``np.`` 

### Function: test_names_are_undersood_by_dtype(self, t)

**Description:** Test the dtype constructor maps names back to the type 

### Function: test_bool_definition(self)
