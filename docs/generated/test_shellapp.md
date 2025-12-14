## AI Summary

A file named test_shellapp.py.


## Class: TestFileToRun

**Description:** Test the behavior of the file_to_run parameter.

### Function: test_py_script_file_attribute(self)

**Description:** Test that `__file__` is set when running `ipython file.py`

### Function: test_ipy_script_file_attribute(self)

**Description:** Test that `__file__` is set when running `ipython file.ipy`

### Function: test_py_script_file_attribute_interactively(self)

**Description:** Test that `__file__` is not set after `ipython -i file.py`
