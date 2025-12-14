## AI Summary

A file named pytest.py.


### Function: execute(callback)

### Function: infer_anonymous_param(func)

### Function: goto_anonymous_param(func)

### Function: complete_param_names(func)

### Function: _goto_pytest_fixture(module_context, name, skip_own_module)

### Function: _is_a_pytest_param_and_inherited(param_name)

**Description:** Pytest params are either in a `test_*` function or have a pytest fixture
with the decorator @pytest.fixture.

This is a heuristic and will work in most cases.

### Function: _is_pytest_func(func_name, decorator_nodes)

### Function: _find_pytest_plugin_modules()

**Description:** Finds pytest plugin modules hooked by setuptools entry points

See https://docs.pytest.org/en/stable/how-to/writing_plugins.html#setuptools-entry-points

### Function: _iter_pytest_modules(module_context, skip_own_module)

### Function: _load_pytest_plugins(module_context, name)

## Class: FixtureFilter

### Function: wrapper(value, arguments)

### Function: get_returns(value)

### Function: wrapper(param_name)

### Function: wrapper(param_name)

### Function: wrapper(context, func_name, decorator_nodes)

### Function: _filter(self, names)

### Function: _is_fixture(self, context, name)
