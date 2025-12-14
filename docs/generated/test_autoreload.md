## AI Summary

A file named test_autoreload.py.


## Class: FakeShell

## Class: Fixture

**Description:** Fixture for creating test module files

### Function: pickle_get_current_class(obj)

**Description:** Original issue comes from pickle; hence the name.

## Class: TestAutoreload

### Function: __init__(self)

### Function: showtraceback(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)

### Function: run_code(self, code)

### Function: push(self, items)

### Function: magic_autoreload(self, parameter)

### Function: magic_aimport(self, parameter, stream)

### Function: setUp(self)

### Function: tearDown(self)

### Function: get_module(self)

### Function: write_file(self, filename, content)

**Description:** Write a file, and force a timestamp difference of at least one second

Notes
-----
Python's .pyc files record the timestamp of their compilation
with a time resolution of one second.

Therefore, we need to force a timestamp difference between .py
and .pyc, without having the .py file be timestamped in the
future, and without changing the timestamp of the .pyc file
(because that is stored in the file).  The only reliable way
to achieve this seems to be to sleep.

### Function: new_module(self, code)

### Function: test_reload_enums(self)

### Function: test_reload_class_type(self)

### Function: test_reload_class_attributes(self)

### Function: test_comparing_numpy_structures(self)

### Function: test_autoload_newly_added_objects(self)

### Function: test_verbose_names(self)

### Function: test_aimport_parsing(self)

### Function: test_autoreload_output(self)

### Function: _check_smoketest(self, use_aimport)

**Description:** Functional test for the automatic reloader using either
'%autoreload 1' or '%autoreload 2'

### Function: test_smoketest_aimport(self)

### Function: test_smoketest_autoreload(self)

## Class: AutoreloadSettings

### Function: gather_settings(mode)

### Function: check_module_contents()

### Function: check_module_contents()
