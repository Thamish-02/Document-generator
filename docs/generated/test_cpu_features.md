## AI Summary

A file named test_cpu_features.py.


### Function: assert_features_equal(actual, desired, fname)

### Function: _text_to_list(txt)

## Class: AbstractTest

## Class: TestEnvPrivation

## Class: Test_X86_Features

## Class: Test_POWER_Features

## Class: Test_ZARCH_Features

## Class: Test_ARM_Features

### Function: load_flags(self)

### Function: test_features(self)

### Function: cpu_have(self, feature_name)

### Function: load_flags_cpuinfo(self, magic_key)

### Function: get_cpuinfo_item(self, magic_key)

### Function: load_flags_auxv(self)

### Function: setup_class(self, tmp_path_factory)

### Function: _run(self)

### Function: _expect_error(self, msg, err_type, no_error_msg)

### Function: setup_method(self)

**Description:** Ensure that the environment is reset

### Function: test_runtime_feature_selection(self)

**Description:** Ensure that when selecting `NPY_ENABLE_CPU_FEATURES`, only the
features exactly specified are dispatched.

### Function: test_both_enable_disable_set(self, enabled, disabled)

**Description:** Ensure that when both environment variables are set then an
ImportError is thrown

### Function: test_variable_too_long(self, action)

**Description:** Test that an error is thrown if the environment variables are too long
to be processed. Current limit is 1024, but this may change later.

### Function: test_impossible_feature_disable(self)

**Description:** Test that a RuntimeError is thrown if an impossible feature-disabling
request is made. This includes disabling a baseline feature.

### Function: test_impossible_feature_enable(self)

**Description:** Test that a RuntimeError is thrown if an impossible feature-enabling
request is made. This includes enabling a feature not supported by the
machine, or disabling a baseline optimization.

### Function: load_flags(self)

### Function: load_flags(self)

### Function: load_flags(self)

### Function: load_flags(self)
