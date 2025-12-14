## AI Summary

A file named gnu.py.


### Function: is_win64()

## Class: GnuFCompiler

## Class: Gnu95FCompiler

### Function: _can_target(cmd, arch)

**Description:** Return true if the architecture supports the -arch flag

### Function: gnu_version_match(self, version_string)

**Description:** Handle the different versions of GNU fortran compilers

### Function: version_match(self, version_string)

### Function: get_flags_linker_so(self)

### Function: get_libgcc_dir(self)

### Function: get_libgfortran_dir(self)

### Function: get_library_dirs(self)

### Function: get_libraries(self)

### Function: get_flags_debug(self)

### Function: get_flags_opt(self)

### Function: _c_arch_flags(self)

**Description:** Return detected arch flags from CFLAGS 

### Function: get_flags_arch(self)

### Function: runtime_library_dir_option(self, dir)

### Function: version_match(self, version_string)

### Function: _universal_flags(self, cmd)

**Description:** Return a list of -arch flags for every supported architecture.

### Function: get_flags(self)

### Function: get_flags_linker_so(self)

### Function: get_library_dirs(self)

### Function: get_libraries(self)

### Function: get_target(self)

### Function: _hash_files(self, filenames)

### Function: _link_wrapper_lib(self, objects, output_dir, extra_dll_dir, chained_dlls, is_archive)

**Description:** Create a wrapper shared library for the given objects

Return an MSVC-compatible lib

### Function: can_ccompiler_link(self, compiler)

### Function: wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir)

**Description:** Convert a set of object files that are not compatible with the default
linker, to a file that is compatible.
