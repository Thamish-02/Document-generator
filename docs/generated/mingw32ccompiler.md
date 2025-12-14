## AI Summary

A file named mingw32ccompiler.py.


### Function: get_msvcr_replacement()

**Description:** Replacement for outdated version of get_msvcr from cygwinccompiler

## Class: Mingw32CCompiler

**Description:** A modified MingW32 compiler compatible with an MSVC built Python.

    

### Function: find_python_dll()

### Function: dump_table(dll)

### Function: generate_def(dll, dfile)

**Description:** Given a dll file location,  get all its exported symbols and dump them
into the given def file.

The .def file will be overwritten

### Function: find_dll(dll_name)

### Function: build_msvcr_library(debug)

### Function: build_import_library()

### Function: _check_for_import_lib()

**Description:** Check if an import library for the Python runtime already exists.

### Function: _build_import_library_amd64()

### Function: _build_import_library_x86()

**Description:** Build the import libraries for Mingw32-gcc on Windows
    

### Function: msvc_manifest_xml(maj, min)

**Description:** Given a major and minor version of the MSVCR, returns the
corresponding XML file.

### Function: manifest_rc(name, type)

**Description:** Return the rc file used to generate the res file which will be embedded
as manifest for given manifest file name, of given type ('dll' or
'exe').

Parameters
----------
name : str
        name of the manifest file to embed
type : str {'dll', 'exe'}
        type of the binary which will embed the manifest

### Function: check_embedded_msvcr_match_linked(msver)

**Description:** msver is the ms runtime version used for the MANIFEST.

### Function: configtest_name(config)

### Function: manifest_name(config)

### Function: rc_name(config)

### Function: generate_manifest(config)

### Function: __init__(self, verbose, dry_run, force)

### Function: link(self, target_desc, objects, output_filename, output_dir, libraries, library_dirs, runtime_library_dirs, export_symbols, debug, extra_preargs, extra_postargs, build_temp, target_lang)

### Function: object_filenames(self, source_filenames, strip_dir, output_dir)

### Function: _find_dll_in_winsxs(dll_name)

### Function: _find_dll_in_path(dll_name)

### Function: get_build_msvc_version()
