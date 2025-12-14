## AI Summary

A file named shimmodule.py.


## Class: ShimWarning

**Description:** A warning to show when a module has moved, and a shim is in its place.

## Class: ShimImporter

**Description:** Import hook for a shim.

This ensures that submodule imports return the real target module,
not a clone that will confuse `is` and `isinstance` checks.

## Class: ShimModule

### Function: __init__(self, src, mirror)

### Function: _mirror_name(self, fullname)

**Description:** get the name of the mirrored module

### Function: find_spec(self, fullname, path, target)

### Function: __init__(self)

### Function: __path__(self)

### Function: __spec__(self)

**Description:** Don't produce __spec__ until requested

### Function: __dir__(self)

### Function: __all__(self)

**Description:** Ensure __all__ is always defined

### Function: __getattr__(self, key)

### Function: __repr__(self)
