## AI Summary

A file named autoreload.py.


## Class: ModuleReloader

### Function: update_function(old, new)

**Description:** Upgrade the code object of a function

### Function: update_instances(old, new)

**Description:** Use garbage collector to find all instances that refer to the old
class definition and update their __class__ to point to the new class
definition

### Function: update_class(old, new)

**Description:** Replace stuff in the __dict__ of a class, and upgrade
method code objects, and add new methods, if any

### Function: update_property(old, new)

**Description:** Replace get/set/del functions of a property

### Function: isinstance2(a, b, typ)

### Function: update_generic(a, b)

## Class: StrongRef

### Function: append_obj(module, d, name, obj, autoload)

### Function: superreload(module, reload, old_objects, shell)

**Description:** Enhanced version of the builtin reload function.

superreload remembers objects previously in the module, and

- upgrades the class dictionary of every old class in the module
- upgrades the code object of every old function and method
- clears the module's namespace before reloading

## Class: AutoreloadMagics

### Function: load_ipython_extension(ip)

**Description:** Load the extension in IPython.

### Function: __init__(self, shell)

### Function: mark_module_skipped(self, module_name)

**Description:** Skip reloading the named module in the future

### Function: mark_module_reloadable(self, module_name)

**Description:** Reload the named module in the future (if it is imported)

### Function: aimport_module(self, module_name)

**Description:** Import a module, and mark it reloadable

Returns
-------
top_module : module
    The imported module if it is top-level, or the top-level
top_name : module
    Name of top_module

### Function: filename_and_mtime(self, module)

### Function: check(self, check_all, do_reload)

**Description:** Check whether some modules need to be reloaded.

### Function: __init__(self, obj)

### Function: __call__(self)

### Function: __init__(self)

### Function: autoreload(self, line)

**Description:** %autoreload => Reload modules automatically

%autoreload or %autoreload now
Reload all modules (except those excluded by %aimport) automatically
now.

%autoreload 0 or %autoreload off
Disable automatic reloading.

%autoreload 1 or %autoreload explicit
Reload only modules imported with %aimport every time before executing
the Python code typed.

%autoreload 2 or %autoreload all
Reload all modules (except those excluded by %aimport) every time
before executing the Python code typed.

%autoreload 3 or %autoreload complete
Same as 2/all, but also but also adds any new objects in the module. See
unit test at IPython/extensions/tests/test_autoreload.py::test_autoload_newly_added_objects

The optional arguments --print and --log control display of autoreload activity. The default
is to act silently; --print (or -p) will print out the names of modules that are being
reloaded, and --log (or -l) outputs them to the log at INFO level.

The optional argument --hide-errors hides any errors that can happen when trying to
reload code.

Reloading Python modules in a reliable way is in general
difficult, and unexpected things may occur. %autoreload tries to
work around common pitfalls by replacing function code objects and
parts of classes previously in the module with new versions. This
makes the following things to work:

- Functions and classes imported via 'from xxx import foo' are upgraded
  to new versions when 'xxx' is reloaded.

- Methods and properties of classes are upgraded on reload, so that
  calling 'c.foo()' on an object 'c' created before the reload causes
  the new code for 'foo' to be executed.

Some of the known remaining caveats are:

- Replacing code objects does not always succeed: changing a @property
  in a class to an ordinary method or a method to a member variable
  can cause problems (but in old objects only).

- Functions that are removed (eg. via monkey-patching) from a module
  before it is reloaded are not upgraded.

- C extension modules cannot be reloaded, and so cannot be
  autoreloaded.

### Function: aimport(self, parameter_s, stream)

**Description:** %aimport => Import modules for automatic reloading.

%aimport
List modules to automatically import and not to import.

%aimport foo
Import module 'foo' and mark it to be autoreloaded for %autoreload explicit

%aimport foo, bar
Import modules 'foo', 'bar' and mark them to be autoreloaded for %autoreload explicit

%aimport -foo, bar
Mark module 'foo' to not be autoreloaded for %autoreload explicit, all, or complete, and 'bar'
to be autoreloaded for mode explicit.

### Function: pre_run_cell(self, info)

### Function: post_execute_hook(self)

**Description:** Cache the modification times of any modules imported in this execution

### Function: pl(msg)
