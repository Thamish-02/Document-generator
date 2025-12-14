## AI Summary

A file named qt_loaders.py.


## Class: ImportDenier

**Description:** Import Hook that will guard against bad Qt imports
once IPython commits to a specific binding

### Function: commit_api(api)

**Description:** Commit to a particular API, and trigger ImportErrors on subsequent
dangerous imports

### Function: loaded_api()

**Description:** Return which API is loaded, if any

If this returns anything besides None,
importing any other Qt binding is unsafe.

Returns
-------
None, 'pyside6', 'pyqt6', 'pyside2', 'pyside', 'pyqt', 'pyqt5', 'pyqtv1'

### Function: has_binding(api)

**Description:** Safely check for PyQt4/5, PySide or PySide2, without importing submodules

Parameters
----------
api : str [ 'pyqtv1' | 'pyqt' | 'pyqt5' | 'pyside' | 'pyside2' | 'pyqtdefault']
    Which module to check for

Returns
-------
True if the relevant module appears to be importable

### Function: qtapi_version()

**Description:** Return which QString API has been set, if any

Returns
-------
The QString API version (1 or 2), or None if not set

### Function: can_import(api)

**Description:** Safely query whether an API is importable, without importing it

### Function: import_pyqt4(version)

**Description:** Import PyQt4

Parameters
----------
version : 1, 2, or None
    Which QString/QVariant API to use. Set to None to use the system
    default
ImportErrors raised within this function are non-recoverable

### Function: import_pyqt5()

**Description:** Import PyQt5

ImportErrors raised within this function are non-recoverable

### Function: import_pyqt6()

**Description:** Import PyQt6

ImportErrors raised within this function are non-recoverable

### Function: import_pyside()

**Description:** Import PySide

ImportErrors raised within this function are non-recoverable

### Function: import_pyside2()

**Description:** Import PySide2

ImportErrors raised within this function are non-recoverable

### Function: import_pyside6()

**Description:** Import PySide6

ImportErrors raised within this function are non-recoverable

### Function: load_qt(api_options)

**Description:** Attempt to import Qt, given a preference list
of permissible bindings

It is safe to call this function multiple times.

Parameters
----------
api_options : List of strings
    The order of APIs to try. Valid items are 'pyside', 'pyside2',
    'pyqt', 'pyqt5', 'pyqtv1' and 'pyqtdefault'

Returns
-------
A tuple of QtCore, QtGui, QtSvg, QT_API
The first three are the Qt modules. The last is the
string indicating which module was loaded.

Raises
------
ImportError, if it isn't possible to import any requested
bindings (either because they aren't installed, or because
an incompatible library has already been installed)

### Function: enum_factory(QT_API, QtCore)

**Description:** Construct an enum helper to account for PyQt5 <-> PyQt6 changes.

### Function: __init__(self)

### Function: forbid(self, module_name)

### Function: find_spec(self, fullname, path, target)

### Function: get_attrs(module)

### Function: _enum(name)
