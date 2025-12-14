## AI Summary

A file named test_public_api.py.


### Function: check_dir(module, module_name)

**Description:** Returns a mapping of all objects with the wrong __module__ attribute.

### Function: test_numpy_namespace()

### Function: test_import_lazy_import(name)

**Description:** Make sure we can actually use the modules we lazy load.

While not exported as part of the public API, it was accessible.  With the
use of __getattr__ and __dir__, this isn't always true It can happen that
an infinite recursion may happen.

This is the only way I found that would force the failure to appear on the
badly implemented code.

We also test for the presence of the lazily imported modules in dir

### Function: test_dir_testing()

**Description:** Assert that output of dir has only one "testing/tester"
attribute without duplicate

### Function: test_numpy_linalg()

### Function: test_numpy_fft()

### Function: test_NPY_NO_EXPORT()

### Function: is_unexpected(name)

**Description:** Check if this needs to be considered.

### Function: test_all_modules_are_expected()

**Description:** Test that we don't add anything that looks like a new public module by
accident.  Check is based on filenames.

### Function: test_all_modules_are_expected_2()

**Description:** Method checking all objects. The pkgutil-based method in
`test_all_modules_are_expected` does not catch imports into a namespace,
only filenames.  So this test is more thorough, and checks this like:

    import .lib.scimath as emath

To check if something in a module is (effectively) public, one can check if
there's anything in that namespace that's a public function/object but is
not exposed in a higher-level namespace.  For example for a `numpy.lib`
submodule::

    mod = np.lib.mixins
    for obj in mod.__all__:
        if obj in np.__all__:
            continue
        elif obj in np.lib.__all__:
            continue

        else:
            print(obj)

### Function: test_api_importable()

**Description:** Check that all submodules listed higher up in this file can be imported

Note that if a PRIVATE_BUT_PRESENT_MODULES entry goes missing, it may
simply need to be removed from the list (deprecation may or may not be
needed - apply common sense).

### Function: test_array_api_entry_point()

**Description:** Entry point for Array API implementation can be found with importlib and
returns the main numpy namespace.

### Function: test_main_namespace_all_dir_coherence()

**Description:** Checks if `dir(np)` and `np.__all__` are consistent and return
the same content, excluding exceptions and private members.

### Function: test_core_shims_coherence()

**Description:** Check that all "semi-public" members of `numpy._core` are also accessible
from `numpy.core` shims.

### Function: test_functions_single_location()

**Description:** Check that each public function is available from one location only.

Test performs BFS search traversing NumPy's public API. It flags
any function-like object that is accessible from more that one place.

### Function: test___module___attribute()

### Function: _check___qualname__(obj)

### Function: test___qualname___attribute()

### Function: find_unexpected_members(mod_name)

### Function: check_importable(module_name)

### Function: _remove_private_members(member_set)

### Function: _remove_exceptions(member_set)
