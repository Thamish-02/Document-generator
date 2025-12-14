## AI Summary

A file named mypy_plugin.py.


### Function: _get_precision_dict()

### Function: _get_extended_precision_list()

### Function: _get_c_intp_name()

### Function: _hook(ctx)

**Description:** Replace a type-alias with a concrete ``NBitBase`` subclass.

### Function: _index(iterable, id)

**Description:** Identify the first ``ImportFrom`` instance the specified `id`.

### Function: _override_imports(file, module, imports)

**Description:** Override the first `module`-based import with new `imports`.

## Class: _NumpyPlugin

**Description:** A mypy plugin for handling versus numpy-specific typing tasks.

### Function: plugin(version)

**Description:** An entry-point for mypy.

### Function: plugin(version)

**Description:** An entry-point for mypy.

### Function: get_type_analyze_hook(self, fullname)

**Description:** Set the precision of platform-specific `numpy.number`
subclasses.

For example: `numpy.int_`, `numpy.longlong` and `numpy.longdouble`.

### Function: get_additional_deps(self, file)

**Description:** Handle all import-based overrides.

* Import platform-specific extended-precision `numpy.number`
  subclasses (*e.g.* `numpy.float96`, `numpy.float128` and
  `numpy.complex256`).
* Import the appropriate `ctypes` equivalent to `numpy.intp`.
