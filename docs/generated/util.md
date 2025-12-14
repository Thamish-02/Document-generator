## AI Summary

A file named util.py.


### Function: check_language(lang, code_snippet)

## Class: CompilerChecker

### Function: has_c_compiler()

### Function: has_f77_compiler()

### Function: has_f90_compiler()

### Function: has_fortran_compiler()

### Function: _cleanup()

### Function: get_module_dir()

### Function: get_temp_module_name()

### Function: _memoize(func)

### Function: build_module(source_files, options, skip, only, module_name)

**Description:** Compile and import a f2py module, built from the given files.

### Function: build_code(source_code, options, skip, only, suffix, module_name)

**Description:** Compile and import Fortran code using f2py.

## Class: SimplifiedMesonBackend

### Function: build_meson(source_files, module_name)

**Description:** Build a module via Meson and import it.

## Class: F2PyTest

### Function: getpath()

### Function: switchdir(path)

### Function: __init__(self)

### Function: check_compilers(self)

### Function: wrapper()

### Function: __init__(self)

### Function: compile(self)

### Function: module_name(self)

### Function: setup_class(cls)

### Function: setup_method(self)
