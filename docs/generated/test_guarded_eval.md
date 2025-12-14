## AI Summary

A file named test_guarded_eval.py.


### Function: create_context(evaluation)

### Function: module_not_installed(module)

### Function: test_external_not_installed()

**Description:** Because attribute check requires checking if object is not of allowed
external type, this tests logic for absence of external module.

### Function: test_external_changed_api(monkeypatch)

**Description:** Check that the execution rejects if external API changed paths

### Function: test_pandas_series_iloc()

### Function: test_rejects_custom_properties()

### Function: test_accepts_non_overriden_properties()

### Function: test_pandas_series()

### Function: test_pandas_bad_series()

### Function: test_pandas_dataframe_loc()

### Function: test_named_tuple()

### Function: test_dict()

### Function: test_set()

### Function: test_list()

### Function: test_dict_literal()

### Function: test_list_literal()

### Function: test_set_literal()

### Function: test_evaluates_if_expression()

### Function: test_object()

### Function: test_number_attributes(code, expected)

### Function: test_method_descriptor()

## Class: HeapType

## Class: CallCreatesHeapType

## Class: CallCreatesBuiltin

## Class: HasStaticMethod

## Class: InitReturnsFrozenset

## Class: StringAnnotation

## Class: TestProtocol

## Class: TestProtocolImplementer

## Class: Movie

## Class: SpecialTyping

### Function: test_evaluates_calls(data, code, expected, equality)

### Function: test_mocks_attributes_of_call_results(data, code, expected_attributes)

### Function: test_mocks_items_of_call_results(data, code, expected_items)

### Function: test_rejects_calls_with_side_effects(data, bad)

### Function: test_evaluates_complex_cases(code, expected, context)

### Function: test_evaluates_literals(code, expected, context)

### Function: test_evaluates_unary_operations(code, expected, context)

### Function: test_evaluates_binary_operations(code, expected, context)

### Function: test_evaluates_comparisons(code, expected, context)

### Function: test_guards_comparisons()

### Function: test_guards_unary_operations()

### Function: test_guards_binary_operations()

### Function: test_guards_attributes()

### Function: test_access_builtins(context)

### Function: test_access_builtins_fails()

### Function: test_rejects_forbidden()

### Function: test_guards_locals_and_globals()

### Function: test_access_locals_and_globals()

### Function: test_rejects_side_effect_syntax(code, context)

### Function: test_subscript()

### Function: test_unbind_method()

### Function: test_assumption_instance_attr_do_not_matter()

**Description:** This is semi-specified in Python documentation.

However, since the specification says 'not guaranteed
to work' rather than 'is forbidden to work', future
versions could invalidate this assumptions. This test
is meant to catch such a change if it ever comes true.

### Function: test_assumption_named_tuples_share_getitem()

**Description:** Check assumption on named tuples sharing __getitem__

### Function: test_module_access()

## Class: Custom

## Class: BadProperty

## Class: GoodProperty

## Class: BadItemSeries

## Class: BadAttrSeries

## Class: GoodNamedTuple

## Class: BadNamedTuple

### Function: __call__(self)

### Function: __call__(self)

### Function: static_method()

### Function: __new__(self)

### Function: heap(self)

### Function: copy(self)

### Function: test_method(self)

### Function: test_method(self)

### Function: custom_int_type(self)

### Function: custom_heap_type(self)

### Function: int_type_alias(self)

### Function: heap_type_alias(self)

### Function: literal(self)

### Function: literal_string(self)

### Function: self(self)

### Function: any_str(self, x)

### Function: annotated(self)

### Function: annotated_self(self)

### Function: int_type_guard(self, x)

### Function: optional_float(self)

### Function: union_str_and_int(self)

### Function: protocol(self)

### Function: typed_dict(self)

## Class: GoodEq

## Class: BadEq

## Class: GoodOp

## Class: BadOpInv

## Class: BadOpInverse

## Class: GoodOp

## Class: BadOp

## Class: GoodAttr

## Class: BadAttr1

## Class: BadAttr2

## Class: X

## Class: T

### Function: f(self)

## Class: A

## Class: B

### Function: __init__(self)

### Function: __getattr__(self, key)

### Function: iloc(self)

### Function: __getitem__(self, key)

### Function: __getattr__(self, key)

### Function: __getitem__(self, key)

### Function: __eq__(self, other)

### Function: __inv__(self, other)

### Function: __inv__(self, other)

### Function: __add__(self, other)

### Function: __getattr__(self, key)

### Function: __getattribute__(self, key)

### Function: index(self, k)

### Function: __getitem__(self, k)

### Function: __getattr__(self, k)
