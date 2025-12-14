## AI Summary

A file named config.py.


## Class: config

## Class: GrabStdout

### Function: initialize_options(self)

### Function: _check_compiler(self)

### Function: _wrap_method(self, mth, lang, args)

### Function: _compile(self, body, headers, include_dirs, lang)

### Function: _link(self, body, headers, include_dirs, libraries, library_dirs, lang)

### Function: check_header(self, header, include_dirs, library_dirs, lang)

### Function: check_decl(self, symbol, headers, include_dirs)

### Function: check_macro_true(self, symbol, headers, include_dirs)

### Function: check_type(self, type_name, headers, include_dirs, library_dirs)

**Description:** Check type availability. Return True if the type can be compiled,
False otherwise

### Function: check_type_size(self, type_name, headers, include_dirs, library_dirs, expected)

**Description:** Check size of a given type.

### Function: check_func(self, func, headers, include_dirs, libraries, library_dirs, decl, call, call_args)

### Function: check_funcs_once(self, funcs, headers, include_dirs, libraries, library_dirs, decl, call, call_args)

**Description:** Check a list of functions at once.

This is useful to speed up things, since all the functions in the funcs
list will be put in one compilation unit.

Arguments
---------
funcs : seq
    list of functions to test
include_dirs : seq
    list of header paths
libraries : seq
    list of libraries to link the code snippet to
library_dirs : seq
    list of library paths
decl : dict
    for every (key, value), the declaration in the value will be
    used for function in key. If a function is not in the
    dictionary, no declaration will be used.
call : dict
    for every item (f, value), if the value is True, a call will be
    done to the function f.

### Function: check_inline(self)

**Description:** Return the inline keyword recognized by the compiler, empty string
otherwise.

### Function: check_restrict(self)

**Description:** Return the restrict keyword recognized by the compiler, empty string
otherwise.

### Function: check_compiler_gcc(self)

**Description:** Return True if the C compiler is gcc

### Function: check_gcc_function_attribute(self, attribute, name)

### Function: check_gcc_function_attribute_with_intrinsics(self, attribute, name, code, include)

### Function: check_gcc_variable_attribute(self, attribute)

### Function: check_gcc_version_at_least(self, major, minor, patchlevel)

**Description:** Return True if the GCC version is greater than or equal to the
specified version.

### Function: get_output(self, body, headers, include_dirs, libraries, library_dirs, lang, use_tee)

**Description:** Try to compile, link to an executable, and run a program
built from 'body' and 'headers'. Returns the exit status code
of the program and its output.

### Function: __init__(self)

### Function: write(self, data)

### Function: flush(self)

### Function: restore(self)
