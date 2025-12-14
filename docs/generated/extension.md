## AI Summary

A file named extension.py.


## Class: Extension

**Description:** Parameters
----------
name : str
    Extension name.
sources : list of str
    List of source file locations relative to the top directory of
    the package.
extra_compile_args : list of str
    Extra command line arguments to pass to the compiler.
extra_f77_compile_args : list of str
    Extra command line arguments to pass to the fortran77 compiler.
extra_f90_compile_args : list of str
    Extra command line arguments to pass to the fortran90 compiler.

### Function: __init__(self, name, sources, include_dirs, define_macros, undef_macros, library_dirs, libraries, runtime_library_dirs, extra_objects, extra_compile_args, extra_link_args, export_symbols, swig_opts, depends, language, f2py_options, module_dirs, extra_c_compile_args, extra_cxx_compile_args, extra_f77_compile_args, extra_f90_compile_args)

### Function: has_cxx_sources(self)

### Function: has_f2py_sources(self)
