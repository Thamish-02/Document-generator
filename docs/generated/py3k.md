## AI Summary

A file named py3k.py.


### Function: asunicode(s)

### Function: asbytes(s)

### Function: asstr(s)

### Function: isfileobj(f)

### Function: open_latin1(filename, mode)

### Function: sixu(s)

### Function: getexception()

### Function: asbytes_nested(x)

### Function: asunicode_nested(x)

### Function: is_pathlib_path(obj)

**Description:** Check whether obj is a `pathlib.Path` object.

Prefer using ``isinstance(obj, os.PathLike)`` instead of this function.

## Class: contextlib_nullcontext

**Description:** Context manager that does no additional processing.

Used as a stand-in for a normal context manager, when a particular
block of code is only sometimes used with a normal context manager:

cm = optional_cm if condition else nullcontext()
with cm:
    # Perform operation, using optional_cm if condition is True

.. note::
    Prefer using `contextlib.nullcontext` instead of this context manager.

### Function: npy_load_module(name, fn, info)

**Description:** Load a module. Uses ``load_module`` which will be deprecated in python
3.12. An alternative that uses ``exec_module`` is in
numpy.distutils.misc_util.exec_mod_from_location

Parameters
----------
name : str
    Full module name.
fn : str
    Path to module file.
info : tuple, optional
    Only here for backward compatibility with Python 2.*.

Returns
-------
mod : module

### Function: __init__(self, enter_result)

### Function: __enter__(self)

### Function: __exit__(self)
