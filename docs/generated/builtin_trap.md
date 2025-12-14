## AI Summary

A file named builtin_trap.py.


## Class: __BuiltinUndefined

## Class: __HideBuiltin

## Class: BuiltinTrap

### Function: __init__(self, shell)

### Function: __enter__(self)

### Function: __exit__(self, type, value, traceback)

### Function: add_builtin(self, key, value)

**Description:** Add a builtin and save the original.

### Function: remove_builtin(self, key, orig)

**Description:** Remove an added builtin and re-set the original.

### Function: activate(self)

**Description:** Store ipython references in the __builtin__ namespace.

### Function: deactivate(self)

**Description:** Remove any builtins which might have been added by add_builtins, or
restore overwritten ones to their previous values.
