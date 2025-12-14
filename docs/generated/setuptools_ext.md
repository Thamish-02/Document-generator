## AI Summary

A file named setuptools_ext.py.


### Function: error(msg)

### Function: execfile(filename, glob)

### Function: add_cffi_module(dist, mod_spec)

### Function: _set_py_limited_api(Extension, kwds)

**Description:** Add py_limited_api to kwds if setuptools >= 26 is in use.
Do not alter the setting if it already exists.
Setuptools takes care of ignoring the flag on Python 2 and PyPy.

CPython itself should ignore the flag in a debugging version
(by not listing .abi3.so in the extensions it supports), but
it doesn't so far, creating troubles.  That's why we check
for "not hasattr(sys, 'gettotalrefcount')" (the 2.7 compatible equivalent
of 'd' not in sys.abiflags). (http://bugs.python.org/issue28401)

On Windows, with CPython <= 3.4, it's better not to use py_limited_api
because virtualenv *still* doesn't copy PYTHON3.DLL on these versions.
Recently (2020) we started shipping only >= 3.5 wheels, though.  So
we'll give it another try and set py_limited_api on Windows >= 3.5.

### Function: _add_c_module(dist, ffi, module_name, source, source_extension, kwds)

### Function: _add_py_module(dist, ffi, module_name)

### Function: cffi_modules(dist, attr, value)

### Function: make_mod(tmpdir, pre_run)

## Class: build_ext_make_mod

### Function: generate_mod(py_file)

## Class: build_py_make_mod

## Class: build_ext_make_mod

### Function: run(self)

### Function: run(self)

### Function: get_source_files(self)

### Function: run(self)
