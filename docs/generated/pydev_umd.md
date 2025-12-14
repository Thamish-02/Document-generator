## AI Summary

A file named pydev_umd.py.


## Class: UserModuleDeleter

**Description:** User Module Deleter (UMD) aims at deleting user modules
to force Python to deeply reload them during import

pathlist [list]: ignore list in terms of module path
namelist [list]: ignore list in terms of module name

### Function: _set_globals_function(get_globals)

### Function: _get_globals()

**Description:** Return current Python interpreter globals namespace

### Function: runfile(filename, args, wdir, namespace)

**Description:** Run filename
args: command line arguments (string)
wdir: working directory

### Function: __init__(self, namelist, pathlist)

### Function: is_module_ignored(self, modname, modpath)

### Function: run(self, verbose)

**Description:** Del user modules to force Python to deeply reload them

Do not del modules which are considered as system modules, i.e.
modules installed in subdirectories of Python interpreter's binary
Do not del C modules
