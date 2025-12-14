## AI Summary

A file named _meson.py.


## Class: MesonTemplate

**Description:** Template meson build file generation class.

## Class: MesonBackend

### Function: _prepare_sources(mname, sources, bdir)

### Function: _get_flags(fc_flags)

### Function: __init__(self, modulename, sources, deps, libraries, library_dirs, include_dirs, object_files, linker_args, fortran_args, build_type, python_exe)

### Function: meson_build_template(self)

### Function: initialize_template(self)

### Function: sources_substitution(self)

### Function: deps_substitution(self)

### Function: libraries_substitution(self)

### Function: include_substitution(self)

### Function: fortran_args_substitution(self)

### Function: generate_meson_build(self)

### Function: __init__(self)

### Function: _move_exec_to_root(self, build_dir)

### Function: write_meson_build(self, build_dir)

**Description:** Writes the meson build file at specified location

### Function: _run_subprocess_command(self, command, cwd)

### Function: run_meson(self, build_dir)

### Function: compile(self)
