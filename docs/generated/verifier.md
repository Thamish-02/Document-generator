## AI Summary

A file named verifier.py.


## Class: Verifier

### Function: _locate_engine_class(ffi, force_generic_engine)

### Function: _caller_dir_pycache()

### Function: set_tmpdir(dirname)

**Description:** Set the temporary directory to use instead of __pycache__.

### Function: cleanup_tmpdir(tmpdir, keep_so)

**Description:** Clean up the temporary directory by removing all files in it
called `_cffi_*.{c,so}` as well as the `build` subdirectory.

### Function: _get_so_suffixes()

### Function: _ensure_dir(filename)

### Function: _extension_suffixes()

### Function: _extension_suffixes()

## Class: NativeIO

### Function: __init__(self, ffi, preamble, tmpdir, modulename, ext_package, tag, force_generic_engine, source_extension, flags, relative_to)

### Function: write_source(self, file)

**Description:** Write the C source code.  It is produced in 'self.sourcefilename',
which can be tweaked beforehand.

### Function: compile_module(self)

**Description:** Write the C source code (if not done already) and compile it.
This produces a dynamic link library in 'self.modulefilename'.

### Function: load_library(self)

**Description:** Get a C module from this Verifier instance.
Returns an instance of a FFILibrary class that behaves like the
objects returned by ffi.dlopen(), but that delegates all
operations to the C module.  If necessary, the C code is written
and compiled first.

### Function: get_module_name(self)

### Function: get_extension(self)

### Function: generates_python_module(self)

### Function: make_relative_to(self, kwds, relative_to)

### Function: _locate_module(self)

### Function: _write_source_to(self, file)

### Function: _write_source(self, file)

### Function: _compile_module(self)

### Function: _load_library(self)

### Function: write(self, s)
