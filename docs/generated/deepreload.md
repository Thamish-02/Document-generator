## AI Summary

A file named deepreload.py.


### Function: replace_import_hook(new_import)

### Function: get_parent(globals, level)

**Description:** parent, name = get_parent(globals, level)

Return the package that an import is being performed in.  If globals comes
from the module foo.bar.bat (not itself a package), this returns the
sys.modules entry for foo.bar.  If globals is from a package's __init__.py,
the package's entry in sys.modules is returned.

If globals doesn't come from a package or a module in a package, or a
corresponding entry is not found in sys.modules, None is returned.

### Function: load_next(mod, altmod, name, buf)

**Description:** mod, name, buf = load_next(mod, altmod, name, buf)

altmod is either None or same as mod

### Function: import_submodule(mod, subname, fullname)

**Description:** m = import_submodule(mod, subname, fullname)

### Function: add_submodule(mod, submod, fullname, subname)

**Description:** mod.{subname} = submod

### Function: ensure_fromlist(mod, fromlist, buf, recursive)

**Description:** Handle 'from module import a, b, c' imports.

### Function: deep_import_hook(name, globals, locals, fromlist, level)

**Description:** Replacement for __import__()

### Function: deep_reload_hook(m)

**Description:** Replacement for reload().

### Function: reload(module, exclude)

**Description:** Recursively reload all modules used in the given module.  Optionally
takes a list of modules to exclude from reloading.  The default exclude
list contains modules listed in sys.builtin_module_names with additional
sys, os.path, builtins and __main__, to prevent, e.g., resetting
display, exception, and io hooks.
