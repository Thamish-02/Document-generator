## AI Summary

A file named test_f2py2e.py.


### Function: compiler_check_f2pycli()

### Function: get_io_paths(fname_inp, mname)

**Description:** Takes in a temporary file for testing and returns the expected output and input paths

Here expected output is essentially one of any of the possible generated
files.

..note::

     Since this does not actually run f2py, none of these are guaranteed to
     exist, and module names are typically incorrect

Parameters
----------
fname_inp : str
            The input filename
mname : str, optional
            The name of the module, untitled by default

Returns
-------
genp : NamedTuple PPaths
        The possible paths which are generated, not all of which exist

### Function: hello_world_f90(tmpdir_factory)

**Description:** Generates a single f90 file for testing

### Function: gh23598_warn(tmpdir_factory)

**Description:** F90 file for testing warnings in gh23598

### Function: gh22819_cli(tmpdir_factory)

**Description:** F90 file for testing disallowed CLI arguments in ghff819

### Function: hello_world_f77(tmpdir_factory)

**Description:** Generates a single f77 file for testing

### Function: retreal_f77(tmpdir_factory)

**Description:** Generates a single f77 file for testing

### Function: f2cmap_f90(tmpdir_factory)

**Description:** Generates a single f90 file for testing

### Function: test_gh22819_cli(capfd, gh22819_cli, monkeypatch)

**Description:** Check that module names are handled correctly
gh-22819
Essentially, the -m name cannot be used to import the module, so the module
named in the .pyf needs to be used instead

CLI :: -m and a .pyf file

### Function: test_gh22819_many_pyf(capfd, gh22819_cli, monkeypatch)

**Description:** Only one .pyf file allowed
gh-22819
CLI :: .pyf files

### Function: test_gh23598_warn(capfd, gh23598_warn, monkeypatch)

### Function: test_gen_pyf(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that a signature file is generated via the CLI
CLI :: -h

### Function: test_gen_pyf_stdout(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that a signature file can be dumped to stdout
CLI :: -h

### Function: test_gen_pyf_no_overwrite(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that the CLI refuses to overwrite signature files
CLI :: -h without --overwrite-signature

### Function: test_untitled_cli(capfd, hello_world_f90, monkeypatch)

**Description:** Check that modules are named correctly

CLI :: defaults

### Function: test_no_py312_distutils_fcompiler(capfd, hello_world_f90, monkeypatch)

**Description:** Check that no distutils imports are performed on 3.12
CLI :: --fcompiler --help-link --backend distutils

### Function: test_f2py_skip(capfd, retreal_f77, monkeypatch)

**Description:** Tests that functions can be skipped
CLI :: skip:

### Function: test_f2py_only(capfd, retreal_f77, monkeypatch)

**Description:** Test that functions can be kept by only:
CLI :: only:

### Function: test_file_processing_switch(capfd, hello_world_f90, retreal_f77, monkeypatch)

**Description:** Tests that it is possible to return to file processing mode
CLI :: :
BUG: numpy-gh #20520

### Function: test_mod_gen_f77(capfd, hello_world_f90, monkeypatch)

**Description:** Checks the generation of files based on a module name
CLI :: -m

### Function: test_mod_gen_gh25263(capfd, hello_world_f77, monkeypatch)

**Description:** Check that pyf files are correctly generated with module structure
CLI :: -m <name> -h pyf_file
BUG: numpy-gh #20520

### Function: test_lower_cmod(capfd, hello_world_f77, monkeypatch)

**Description:** Lowers cases by flag or when -h is present

CLI :: --[no-]lower

### Function: test_lower_sig(capfd, hello_world_f77, monkeypatch)

**Description:** Lowers cases in signature files by flag or when -h is present

CLI :: --[no-]lower -h

### Function: test_build_dir(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that the build directory can be specified

CLI :: --build-dir

### Function: test_overwrite(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that the build directory can be specified

CLI :: --overwrite-signature

### Function: test_latexdoc(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that TeX documentation is written out

CLI :: --latex-doc

### Function: test_nolatexdoc(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that TeX documentation is written out

CLI :: --no-latex-doc

### Function: test_shortlatex(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that truncated documentation is written out

TODO: Test to ensure this has no effect without --latex-doc
CLI :: --latex-doc --short-latex

### Function: test_restdoc(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that RsT documentation is written out

CLI :: --rest-doc

### Function: test_norestexdoc(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that TeX documentation is written out

CLI :: --no-rest-doc

### Function: test_debugcapi(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that debugging wrappers are written

CLI :: --debug-capi

### Function: test_debugcapi_bld(hello_world_f90, monkeypatch)

**Description:** Ensures that debugging wrappers work

CLI :: --debug-capi -c

### Function: test_wrapfunc_def(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that fortran subroutine wrappers for F77 are included by default

CLI :: --[no]-wrap-functions

### Function: test_nowrapfunc(capfd, hello_world_f90, monkeypatch)

**Description:** Ensures that fortran subroutine wrappers for F77 can be disabled

CLI :: --no-wrap-functions

### Function: test_inclheader(capfd, hello_world_f90, monkeypatch)

**Description:** Add to the include directories

CLI :: -include
TODO: Document this in the help string

### Function: test_inclpath()

**Description:** Add to the include directories

CLI :: --include-paths

### Function: test_hlink()

**Description:** Add to the include directories

CLI :: --help-link

### Function: test_f2cmap(capfd, f2cmap_f90, monkeypatch)

**Description:** Check that Fortran-to-Python KIND specs can be passed

CLI :: --f2cmap

### Function: test_quiet(capfd, hello_world_f90, monkeypatch)

**Description:** Reduce verbosity

CLI :: --quiet

### Function: test_verbose(capfd, hello_world_f90, monkeypatch)

**Description:** Increase verbosity

CLI :: --verbose

### Function: test_version(capfd, monkeypatch)

**Description:** Ensure version

CLI :: -v

### Function: test_npdistop(hello_world_f90, monkeypatch)

**Description:** CLI :: -c

### Function: test_no_freethreading_compatible(hello_world_f90, monkeypatch)

**Description:** CLI :: --no-freethreading-compatible

### Function: test_freethreading_compatible(hello_world_f90, monkeypatch)

**Description:** CLI :: --freethreading_compatible

### Function: test_npd_fcompiler()

**Description:** CLI :: -c --fcompiler

### Function: test_npd_compiler()

**Description:** CLI :: -c --compiler

### Function: test_npd_help_fcompiler()

**Description:** CLI :: -c --help-fcompiler

### Function: test_npd_f77exec()

**Description:** CLI :: -c --f77exec

### Function: test_npd_f90exec()

**Description:** CLI :: -c --f90exec

### Function: test_npd_f77flags()

**Description:** CLI :: -c --f77flags

### Function: test_npd_f90flags()

**Description:** CLI :: -c --f90flags

### Function: test_npd_opt()

**Description:** CLI :: -c --opt

### Function: test_npd_arch()

**Description:** CLI :: -c --arch

### Function: test_npd_noopt()

**Description:** CLI :: -c --noopt

### Function: test_npd_noarch()

**Description:** CLI :: -c --noarch

### Function: test_npd_debug()

**Description:** CLI :: -c --debug

### Function: test_npd_link_auto()

**Description:** CLI :: -c --link-<resource>

### Function: test_npd_lib()

**Description:** CLI :: -c -L/path/to/lib/ -l<libname>

### Function: test_npd_define()

**Description:** CLI :: -D<define>

### Function: test_npd_undefine()

**Description:** CLI :: -U<name>

### Function: test_npd_incl()

**Description:** CLI :: -I/path/to/include/

### Function: test_npd_linker()

**Description:** CLI :: <filename>.o <filename>.so <filename>.a
