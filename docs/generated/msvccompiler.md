## AI Summary

A file named msvccompiler.py.


### Function: _merge(old, new)

**Description:** Concatenate two environment paths avoiding repeats.

Here `old` is the environment string before the base class initialize
function is called and `new` is the string after the call. The new string
will be a fixed string if it is not obtained from the current environment,
or the same as the old string if obtained from the same environment. The aim
here is not to append the new string if it is already contained in the old
string so as to limit the growth of the environment string.

Parameters
----------
old : string
    Previous environment string.
new : string
    New environment string.

Returns
-------
ret : string
    Updated environment string.

## Class: MSVCCompiler

### Function: lib_opts_if_msvc(build_cmd)

**Description:** Add flags if we are using MSVC compiler

We can't see `build_cmd` in our scope, because we have not initialized
the distutils build command, so use this deferred calculation to run
when we are building the library.

### Function: __init__(self, verbose, dry_run, force)

### Function: initialize(self)
