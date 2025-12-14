## AI Summary

A file named parse.py.


### Function: parse_distributions_h(ffi, inc_dir)

**Description:** Parse distributions.h located in inc_dir for CFFI, filling in the ffi.cdef

Read the function declarations without the "#define ..." macros that will
be filled in when loading the library.
