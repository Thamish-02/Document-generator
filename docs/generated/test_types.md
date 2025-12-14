## AI Summary

A file named test_types.py.


### Function: equals_2(checker, instance)

### Function: is_namedtuple(instance)

### Function: is_object_or_named_tuple(checker, instance)

## Class: TestTypeChecker

## Class: TestCustomTypes

### Function: test_is_type(self)

### Function: test_is_unknown_type(self)

### Function: test_checks_can_be_added_at_init(self)

### Function: test_redefine_existing_type(self)

### Function: test_remove(self)

### Function: test_remove_unknown_type(self)

### Function: test_redefine_many(self)

### Function: test_remove_multiple(self)

### Function: test_type_check_can_raise_key_error(self)

**Description:** Make sure no one writes:

    try:
        self._type_checkers[type](...)
    except KeyError:

ignoring the fact that the function itself can raise that.

### Function: test_repr(self)

### Function: test_simple_type_can_be_extended(self)

### Function: test_object_can_be_extended(self)

### Function: test_object_extensions_require_custom_validators(self)

### Function: test_object_extensions_can_handle_custom_validators(self)

### Function: test_unknown_type(self)

### Function: raises_keyerror(checker, instance)

### Function: int_or_str_int(checker, instance)

### Function: coerce_named_tuple(fn)

### Function: coerced(validator, value, instance, schema)
