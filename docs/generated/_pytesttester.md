## AI Summary

A file named _pytesttester.py.


### Function: _show_numpy_info()

## Class: PytestTester

**Description:** Pytest test runner.

A test function is typically added to a package's __init__.py like so::

  from numpy._pytesttester import PytestTester
  test = PytestTester(__name__).test
  del PytestTester

Calling this test function finds and runs all tests associated with the
module and all its sub-modules.

Attributes
----------
module_name : str
    Full path to the package to test.

Parameters
----------
module_name : module name
    The name of the module to test.

Notes
-----
Unlike the previous ``nose``-based implementation, this class is not
publicly exposed as it performs some ``numpy``-specific warning
suppression.

### Function: __init__(self, module_name)

### Function: __call__(self, label, verbose, extra_argv, doctests, coverage, durations, tests)

**Description:** Run tests for module using pytest.

Parameters
----------
label : {'fast', 'full'}, optional
    Identifies the tests to run. When set to 'fast', tests decorated
    with `pytest.mark.slow` are skipped, when 'full', the slow marker
    is ignored.
verbose : int, optional
    Verbosity value for test outputs, in the range 1-3. Default is 1.
extra_argv : list, optional
    List with any extra arguments to pass to pytests.
doctests : bool, optional
    .. note:: Not supported
coverage : bool, optional
    If True, report coverage of NumPy code. Default is False.
    Requires installation of (pip) pytest-cov.
durations : int, optional
    If < 0, do nothing, If 0, report time of all tests, if > 0,
    report the time of the slowest `timer` tests. Default is -1.
tests : test or list of tests
    Tests to be executed with pytest '--pyargs'

Returns
-------
result : bool
    Return True on success, false otherwise.

Notes
-----
Each NumPy module exposes `test` in its namespace to run all tests for
it. For example, to run all tests for numpy.lib:

>>> np.lib.test() #doctest: +SKIP

Examples
--------
>>> result = np.lib.test() #doctest: +SKIP
...
1023 passed, 2 skipped, 6 deselected, 1 xfailed in 10.39 seconds
>>> result
True
