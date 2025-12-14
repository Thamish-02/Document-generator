## AI Summary

A file named __config__.py.


## Class: DisplayModes

### Function: _cleanup(d)

**Description:** Removes empty values in a `dict` recursively
This ensures we remove values that Meson could not provide to CONFIG

### Function: _check_pyyaml()

### Function: show(mode)

**Description:** Show libraries and system information on which NumPy was built
and is being used

Parameters
----------
mode : {`'stdout'`, `'dicts'`}, optional.
    Indicates how to display the config information.
    `'stdout'` prints to console, `'dicts'` returns a dictionary
    of the configuration.

Returns
-------
out : {`dict`, `None`}
    If mode is `'dicts'`, a dict is returned, else None

See Also
--------
get_include : Returns the directory containing NumPy C
              header files.

Notes
-----
1. The `'stdout'` mode will give more readable
   output if ``pyyaml`` is installed

### Function: show_config(mode)
