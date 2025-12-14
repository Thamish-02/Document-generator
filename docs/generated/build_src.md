## AI Summary

A file named build_src.py.


### Function: subst_vars(target, source, d)

**Description:** Substitute any occurrence of @foo@ by d['foo'] from source file into
target.

## Class: build_src

### Function: get_swig_target(source)

### Function: get_swig_modulename(source)

### Function: _find_swig_target(target_dir, name)

### Function: get_f2py_modulename(source)

### Function: initialize_options(self)

### Function: finalize_options(self)

### Function: run(self)

### Function: build_sources(self)

### Function: build_data_files_sources(self)

### Function: _build_npy_pkg_config(self, info, gd)

### Function: build_npy_pkg_config(self)

### Function: build_py_modules_sources(self)

### Function: build_library_sources(self, lib_name, build_info)

### Function: build_extension_sources(self, ext)

### Function: generate_sources(self, sources, extension)

### Function: filter_py_files(self, sources)

### Function: filter_h_files(self, sources)

### Function: filter_files(self, sources, exts)

### Function: template_sources(self, sources, extension)

### Function: pyrex_sources(self, sources, extension)

**Description:** Pyrex not supported; this remains for Cython support (see below)

### Function: generate_a_pyrex_source(self, base, ext_name, source, extension)

**Description:** Pyrex is not supported, but some projects monkeypatch this method.

That allows compiling Cython code, see gh-6955.
This method will remain here for compatibility reasons.

### Function: f2py_sources(self, sources, extension)

### Function: swig_sources(self, sources, extension)
