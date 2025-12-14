## AI Summary

A file named extbuild.py.


### Function: build_and_import_extension(modname, functions)

**Description:** Build and imports a c-extension module `modname` from a list of function
fragments `functions`.


Parameters
----------
functions : list of fragments
    Each fragment is a sequence of func_name, calling convention, snippet.
prologue : string
    Code to precede the rest, usually extra ``#include`` or ``#define``
    macros.
build_dir : pathlib.Path
    Where to build the module, usually a temporary directory
include_dirs : list
    Extra directories to find include files when compiling
more_init : string
    Code to appear in the module PyMODINIT_FUNC

Returns
-------
out: module
    The module will have been loaded and is ready for use

Examples
--------
>>> functions = [("test_bytes", "METH_O", """
    if ( !PyBytesCheck(args)) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
""")]
>>> mod = build_and_import_extension("testme", functions)
>>> assert not mod.test_bytes('abc')
>>> assert mod.test_bytes(b'abc')

### Function: compile_extension_module(name, builddir, include_dirs, source_string, libraries, library_dirs)

**Description:** Build an extension module and return the filename of the resulting
native code file.

Parameters
----------
name : string
    name of the module, possibly including dots if it is a module inside a
    package.
builddir : pathlib.Path
    Where to build the module, usually a temporary directory
include_dirs : list
    Extra directories to find include files when compiling
libraries : list
    Libraries to link into the extension module
library_dirs: list
    Where to find the libraries, ``-L`` passed to the linker

### Function: _convert_str_to_file(source, dirname)

**Description:** Helper function to create a file ``source.c`` in `dirname` that contains
the string in `source`. Returns the file name

### Function: _make_methods(functions, modname)

**Description:** Turns the name, signature, code in functions into complete functions
and lists them in a methods_table. Then turns the methods_table into a
``PyMethodDef`` structure and returns the resulting code fragment ready
for compilation

### Function: _make_source(name, init, body)

**Description:** Combines the code fragments into source code ready to be compiled
    

### Function: _c_compile(cfile, outputfilename, include_dirs, libraries, library_dirs)

### Function: build(cfile, outputfilename, compile_extra, link_extra, include_dirs, libraries, library_dirs)

**Description:** use meson to build

### Function: get_so_suffix()
