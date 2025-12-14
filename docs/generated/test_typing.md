## AI Summary

A file named test_typing.py.


### Function: _key_func(key)

**Description:** Split at the first occurrence of the ``:`` character.

Windows drive-letters (*e.g.* ``C:``) are ignored herein.

### Function: _strip_filename(msg)

**Description:** Strip the filename and line number from a mypy message.

### Function: strip_func(match)

**Description:** `re.sub` helper function for stripping module names.

### Function: run_mypy()

**Description:** Clears the cache and run mypy before running any of the typing tests.

The mypy results are cached in `OUTPUT_MYPY` for further use.

The cache refresh can be skipped using

NUMPY_TYPING_TEST_CLEAR_CACHE=0 pytest numpy/typing/tests

### Function: get_test_cases(directory)

### Function: test_success(path)

### Function: test_fail(path)

### Function: _test_fail(path, expression, error, expected_error, lineno)

### Function: test_reveal(path)

**Description:** Validate that mypy correctly infers the return-types of
the expressions in `path`.

### Function: test_code_runs(path)

**Description:** Validate that the code in `path` properly during runtime.

### Function: test_extended_precision()
