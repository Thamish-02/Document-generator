## AI Summary

A file named pkgconfig.py.


### Function: merge_flags(cfg1, cfg2)

**Description:** Merge values from cffi config flags cfg2 to cf1

Example:
    merge_flags({"libraries": ["one"]}, {"libraries": ["two"]})
    {"libraries": ["one", "two"]}

### Function: call(libname, flag, encoding)

**Description:** Calls pkg-config and returns the output if found
    

### Function: flags_from_pkgconfig(libs)

**Description:** Return compiler line flags for FFI.set_source based on pkg-config output

Usage
    ...
    ffibuilder.set_source("_foo", pkgconfig = ["libfoo", "libbar >= 1.8.3"])

If pkg-config is installed on build machine, then arguments include_dirs,
library_dirs, libraries, define_macros, extra_compile_args and
extra_link_args are extended with an output of pkg-config for libfoo and
libbar.

Raises PkgConfigError in case the pkg-config call fails.

### Function: get_include_dirs(string)

### Function: get_library_dirs(string)

### Function: get_libraries(string)

### Function: get_macros(string)

### Function: get_other_cflags(string)

### Function: get_other_libs(string)

### Function: kwargs(libname)

### Function: _macro(x)
