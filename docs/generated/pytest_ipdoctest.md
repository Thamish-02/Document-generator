## AI Summary

A file named pytest_ipdoctest.py.


### Function: pytest_addoption(parser)

### Function: pytest_unconfigure()

### Function: pytest_collect_file(file_path, parent)

### Function: _is_setup_py(path)

### Function: _is_ipdoctest(config, path, parent)

### Function: _is_main_py(path)

## Class: ReprFailDoctest

## Class: MultipleDoctestFailures

### Function: _init_runner_class()

### Function: _get_runner(checker, verbose, optionflags, continue_on_failure)

## Class: IPDoctestItem

### Function: _get_flag_lookup()

### Function: get_optionflags(parent)

### Function: _get_continue_on_failure(config)

## Class: IPDoctestTextfile

### Function: _check_all_skipped(test)

**Description:** Raise pytest.skip() if all examples in the given DocTest have the SKIP
option set.

### Function: _is_mocked(obj)

**Description:** Return if an object is possibly a mock object by checking the
existence of a highly improbable attribute.

### Function: _patch_unwrap_mock_aware()

**Description:** Context manager which replaces ``inspect.unwrap`` with a version
that's aware of mock objects and doesn't recurse into them.

## Class: IPDoctestModule

### Function: _setup_fixtures(doctest_item)

**Description:** Used by IPDoctestTextfile and IPDoctestItem to setup fixture information.

### Function: _init_checker_class()

### Function: _get_checker()

**Description:** Return a IPDoctestOutputChecker subclass that supports some
additional options:

* ALLOW_UNICODE and ALLOW_BYTES options to ignore u'' and b''
  prefixes (respectively) in string literals. Useful when the same
  ipdoctest should run in Python 2 and Python 3.

* NUMBER to ignore floating-point differences smaller than the
  precision of the literal number in the ipdoctest.

An inner class is used to avoid importing "ipdoctest" at the module
level.

### Function: _get_allow_unicode_flag()

**Description:** Register and return the ALLOW_UNICODE flag.

### Function: _get_allow_bytes_flag()

**Description:** Register and return the ALLOW_BYTES flag.

### Function: _get_number_flag()

**Description:** Register and return the NUMBER flag.

### Function: _get_report_choice(key)

**Description:** Return the actual `ipdoctest` module flag value.

We want to do it as late as possible to avoid importing `ipdoctest` and all
its dependencies when parsing options, as it adds overhead and breaks tests.

### Function: ipdoctest_namespace()

**Description:** Fixture that returns a :py:class:`dict` that will be injected into the
namespace of ipdoctests.

### Function: pytest_collect_file(path, parent)

### Function: import_path(path, root)

### Function: __init__(self, reprlocation_lines)

### Function: toterminal(self, tw)

### Function: __init__(self, failures)

## Class: PytestDoctestRunner

**Description:** Runner to collect failures.

Note that the out variable in this case is a list instead of a
stdout-like object.

### Function: __init__(self, name, parent, runner, dtest)

### Function: from_parent(cls, parent)

**Description:** The public named constructor.

### Function: setup(self)

### Function: teardown(self)

### Function: runtest(self)

### Function: _disable_output_capturing_for_darwin(self)

**Description:** Disable output capturing. Otherwise, stdout is lost to ipdoctest (pytest#985).

### Function: repr_failure(self, excinfo)

### Function: reportinfo(self)

### Function: collect(self)

### Function: _mock_aware_unwrap(func)

### Function: collect(self)

### Function: func()

## Class: LiteralsOutputChecker

### Function: __init__(self, checker, verbose, optionflags, continue_on_failure)

### Function: report_failure(self, out, test, example, got)

### Function: report_unexpected_exception(self, out, test, example, exc_info)

### Function: path(self)

### Function: path(self)

### Function: from_parent(cls, parent)

## Class: MockAwareDocTestFinder

**Description:** A hackish ipdoctest finder that overrides stdlib internals to fix a stdlib bug.

https://github.com/pytest-dev/pytest/issues/3456
https://bugs.python.org/issue25532

### Function: path(self)

### Function: from_parent(cls, parent)

### Function: check_output(self, want, got, optionflags)

### Function: _remove_unwanted_precision(self, want, got)

### Function: _find_lineno(self, obj, source_lines)

**Description:** Doctest code does not take into account `@property`, this
is a hackish way to fix it. https://bugs.python.org/issue17446

Wrapped Doctests will need to be unwrapped so the correct
line number is returned. This will be reported upstream. #8796

### Function: _find(self, tests, obj, name, module, source_lines, globs, seen)

### Function: remove_prefixes(regex, txt)
