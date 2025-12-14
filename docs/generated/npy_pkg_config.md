## AI Summary

A file named npy_pkg_config.py.


## Class: FormatError

**Description:** Exception thrown when there is a problem parsing a configuration file.

## Class: PkgNotFound

**Description:** Exception raised when a package can not be located.

### Function: parse_flags(line)

**Description:** Parse a line from a config file containing compile flags.

Parameters
----------
line : str
    A single line containing one or more compile flags.

Returns
-------
d : dict
    Dictionary of parsed flags, split into relevant categories.
    These categories are the keys of `d`:

    * 'include_dirs'
    * 'library_dirs'
    * 'libraries'
    * 'macros'
    * 'ignored'

### Function: _escape_backslash(val)

## Class: LibraryInfo

**Description:** Object containing build information about a library.

Parameters
----------
name : str
    The library name.
description : str
    Description of the library.
version : str
    Version string.
sections : dict
    The sections of the configuration file for the library. The keys are
    the section headers, the values the text under each header.
vars : class instance
    A `VariableSet` instance, which contains ``(name, value)`` pairs for
    variables defined in the configuration file for the library.
requires : sequence, optional
    The required libraries for the library to be installed.

Notes
-----
All input parameters (except "sections" which is a method) are available as
attributes of the same name.

## Class: VariableSet

**Description:** Container object for the variables defined in a config file.

`VariableSet` can be used as a plain dictionary, with the variable names
as keys.

Parameters
----------
d : dict
    Dict of items in the "variables" section of the configuration file.

### Function: parse_meta(config)

### Function: parse_variables(config)

### Function: parse_sections(config)

### Function: pkg_to_filename(pkg_name)

### Function: parse_config(filename, dirs)

### Function: _read_config_imp(filenames, dirs)

### Function: read_config(pkgname, dirs)

**Description:** Return library info for a package from its configuration file.

Parameters
----------
pkgname : str
    Name of the package (should match the name of the .ini file, without
    the extension, e.g. foo for the file foo.ini).
dirs : sequence, optional
    If given, should be a sequence of directories - usually including
    the NumPy base directory - where to look for npy-pkg-config files.

Returns
-------
pkginfo : class instance
    The `LibraryInfo` instance containing the build information.

Raises
------
PkgNotFound
    If the package is not found.

See Also
--------
misc_util.get_info, misc_util.get_pkg_info

Examples
--------
>>> npymath_info = np.distutils.npy_pkg_config.read_config('npymath')
>>> type(npymath_info)
<class 'numpy.distutils.npy_pkg_config.LibraryInfo'>
>>> print(npymath_info)
Name: npymath
Description: Portable, core math library implementing C99 standard
Requires:
Version: 0.1  #random

### Function: __init__(self, msg)

### Function: __str__(self)

### Function: __init__(self, msg)

### Function: __str__(self)

### Function: __init__(self, name, description, version, sections, vars, requires)

### Function: sections(self)

**Description:** Return the section headers of the config file.

Parameters
----------
None

Returns
-------
keys : list of str
    The list of section headers.

### Function: cflags(self, section)

### Function: libs(self, section)

### Function: __str__(self)

### Function: __init__(self, d)

### Function: _init_parse(self)

### Function: _init_parse_var(self, name, value)

### Function: interpolate(self, value)

### Function: variables(self)

**Description:** Return the list of variable names.

Parameters
----------
None

Returns
-------
names : list of str
    The names of all variables in the `VariableSet` instance.

### Function: __getitem__(self, name)

### Function: __setitem__(self, name, value)

### Function: _read_config(f)

### Function: _interpolate(value)
