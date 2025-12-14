## AI Summary

A file named pydevd_reload.py.


### Function: write_err()

### Function: notify_info0()

### Function: notify_info()

### Function: notify_info2()

### Function: notify_error()

### Function: code_objects_equal(code0, code1)

### Function: xreload(mod)

**Description:** Reload a module in place, updating classes, methods and functions.

mod: a module object

Returns a boolean indicating whether a change was done.

## Class: Reload

### Function: __init__(self, mod, mod_name, mod_filename)

### Function: apply(self)

### Function: _handle_namespace(self, namespace, is_class_namespace)

### Function: _update(self, namespace, name, oldobj, newobj, is_class_namespace)

**Description:** Update oldobj, if possible in place, with newobj.

If oldobj is immutable, this simply returns newobj.

Args:
  oldobj: the object to be updated
  newobj: the object used as the source for the update

### Function: _update_function(self, oldfunc, newfunc)

**Description:** Update a function object.

### Function: _update_method(self, oldmeth, newmeth)

**Description:** Update a method object.

### Function: _update_class(self, oldclass, newclass)

**Description:** Update a class object.

### Function: _update_classmethod(self, oldcm, newcm)

**Description:** Update a classmethod update.

### Function: _update_staticmethod(self, oldsm, newsm)

**Description:** Update a staticmethod update.
