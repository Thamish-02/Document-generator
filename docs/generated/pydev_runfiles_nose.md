## AI Summary

A file named pydev_runfiles_nose.py.


## Class: PydevPlugin

### Function: start_pydev_nose_plugin_singleton(configuration)

### Function: new_consolidate(self, result, batch_result)

**Description:** Used so that it can work with the multiprocess plugin.
Monkeypatched because nose seems a bit unsupported at this time (ideally
the plugin would have this support by default).

### Function: __init__(self, configuration)

### Function: begin(self)

### Function: finalize(self, result)

## Class: Sentinel

### Function: _without_user_address(self, test)

### Function: _get_test_address(self, test)

### Function: report_cond(self, cond, test, captured_output, error)

**Description:** @param cond: fail, error, ok

### Function: startTest(self, test)

### Function: get_io_from_error(self, err)

### Function: get_captured_output(self, test)

### Function: addError(self, test, err)

### Function: addFailure(self, test, err)

### Function: addSuccess(self, test)
