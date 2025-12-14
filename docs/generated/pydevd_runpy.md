## AI Summary

A file named pydevd_runpy.py.


### Function: pkgutil_get_importer(path_item)

**Description:** Retrieve a finder for the given path item

The returned finder is cached in sys.path_importer_cache
if it was newly created by a path hook.

The cache (or part of it) can be cleared manually if a
rescan of sys.path_hooks is necessary.

## Class: _TempModule

**Description:** Temporarily replace a module in sys.modules with an empty namespace

## Class: _ModifiedArgv0

### Function: _run_code(code, run_globals, init_globals, mod_name, mod_spec, pkg_name, script_name)

**Description:** Helper to run code in nominated namespace

### Function: _run_module_code(code, init_globals, mod_name, mod_spec, pkg_name, script_name)

**Description:** Helper to run code in new namespace with sys modified

### Function: _get_module_details(mod_name, error)

## Class: _Error

**Description:** Error that _run_module_as_main() should report without a traceback

### Function: _run_module_as_main(mod_name, alter_argv)

**Description:** Runs the designated module in the __main__ namespace

Note that the executed module will have full access to the
__main__ namespace. If this is not desirable, the run_module()
function should be used to run the module code in a fresh namespace.

At the very least, these variables in __main__ will be overwritten:
    __name__
    __file__
    __cached__
    __loader__
    __package__

### Function: run_module(mod_name, init_globals, run_name, alter_sys)

**Description:** Execute a module's code without importing it

Returns the resulting top level namespace dictionary

### Function: _get_main_module_details(error)

### Function: _get_code_from_file(run_name, fname)

### Function: run_path(path_name, init_globals, run_name)

**Description:** Execute code located at the specified filesystem location

Returns the resulting top level namespace dictionary

The file path may refer directly to a Python script (i.e.
one that could be directly executed with execfile) or else
it may refer to a zipfile or directory containing a top
level __main__.py script.

### Function: __init__(self, mod_name)

### Function: __enter__(self)

### Function: __exit__(self)

### Function: __init__(self, value)

### Function: __enter__(self)

### Function: __exit__(self)
