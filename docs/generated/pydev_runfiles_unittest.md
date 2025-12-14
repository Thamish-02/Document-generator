## AI Summary

A file named pydev_runfiles_unittest.py.


## Class: PydevTextTestRunner

## Class: PydevTestResult

## Class: PydevTestSuite

### Function: _makeResult(self)

### Function: addSubTest(self, test, subtest, err)

**Description:** Called at the end of a subtest.
'err' is None if the subtest ended successfully, otherwise it's a
tuple of values as returned by sys.exc_info().

### Function: startTest(self, test)

### Function: get_test_name(self, test)

### Function: stopTest(self, test)

### Function: _reportErrors(self, errors, failures, captured_output, test_name, diff_time)

### Function: addError(self, test, err)

### Function: addFailure(self, test, err)
