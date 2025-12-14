## AI Summary

A file named misc_util.py.


### Function: clean_up_temporary_directory()

## Class: InstallableLib

**Description:** Container to hold information on an installable library.

Parameters
----------
name : str
    Name of the installed library.
build_info : dict
    Dictionary holding build information.
target_dir : str
    Absolute path specifying where to install the library.

See Also
--------
Configuration.add_installed_library

Notes
-----
The three parameters are stored as attributes with the same names.

### Function: get_num_build_jobs()

**Description:** Get number of parallel build jobs set by the --parallel command line
argument of setup.py
If the command did not receive a setting the environment variable
NPY_NUM_BUILD_JOBS is checked. If that is unset, return the number of
processors on the system, with a maximum of 8 (to prevent
overloading the system if there a lot of CPUs).

Returns
-------
out : int
    number of parallel jobs that can be run

### Function: quote_args(args)

**Description:** Quote list of arguments.

.. deprecated:: 1.22.

### Function: allpath(name)

**Description:** Convert a /-separated pathname to one using the OS's path separator.

### Function: rel_path(path, parent_path)

**Description:** Return path relative to parent_path.

### Function: get_path_from_frame(frame, parent_path)

**Description:** Return path of the module given a frame object from the call stack.

Returned path is relative to parent_path when given,
otherwise it is absolute path.

### Function: njoin()

**Description:** Join two or more pathname components +
- convert a /-separated pathname to one using the OS's path separator.
- resolve `..` and `.` from path.

Either passing n arguments as in njoin('a','b'), or a sequence
of n names as in njoin(['a','b']) is handled, or a mixture of such arguments.

### Function: get_mathlibs(path)

**Description:** Return the MATHLIB line from numpyconfig.h
    

### Function: minrelpath(path)

**Description:** Resolve `..` and '.' from path.
    

### Function: sorted_glob(fileglob)

**Description:** sorts output of python glob for https://bugs.python.org/issue30461
to allow extensions to have reproducible build results

### Function: _fix_paths(paths, local_path, include_non_existing)

### Function: gpaths(paths, local_path, include_non_existing)

**Description:** Apply glob to paths and prepend local_path if needed.
    

### Function: make_temp_file(suffix, prefix, text)

### Function: terminal_has_colors()

### Function: default_text(s)

### Function: red_text(s)

### Function: green_text(s)

### Function: yellow_text(s)

### Function: cyan_text(s)

### Function: blue_text(s)

### Function: cyg2win32(path)

**Description:** Convert a path from Cygwin-native to Windows-native.

Uses the cygpath utility (part of the Base install) to do the
actual conversion.  Falls back to returning the original path if
this fails.

Handles the default ``/cygdrive`` mount prefix as well as the
``/proc/cygdrive`` portable prefix, custom cygdrive prefixes such
as ``/`` or ``/mnt``, and absolute paths such as ``/usr/src/`` or
``/home/username``

Parameters
----------
path : str
   The path to convert

Returns
-------
converted_path : str
    The converted path

Notes
-----
Documentation for cygpath utility:
https://cygwin.com/cygwin-ug-net/cygpath.html
Documentation for the C function it wraps:
https://cygwin.com/cygwin-api/func-cygwin-conv-path.html

### Function: mingw32()

**Description:** Return true when using mingw32 environment.
    

### Function: msvc_runtime_version()

**Description:** Return version of MSVC runtime library, as defined by __MSC_VER__ macro

### Function: msvc_runtime_library()

**Description:** Return name of MSVC runtime library if Python was built with MSVC >= 7

### Function: msvc_runtime_major()

**Description:** Return major version of MSVC runtime coded like get_build_msvc_version

### Function: _get_f90_modules(source)

**Description:** Return a list of Fortran f90 module names that
given source file defines.

### Function: is_string(s)

### Function: all_strings(lst)

**Description:** Return True if all items in lst are string objects. 

### Function: is_sequence(seq)

### Function: is_glob_pattern(s)

### Function: as_list(seq)

### Function: get_language(sources)

**Description:** Determine language value (c,f77,f90) from sources 

### Function: has_f_sources(sources)

**Description:** Return True if sources contains Fortran files 

### Function: has_cxx_sources(sources)

**Description:** Return True if sources contains C++ files 

### Function: filter_sources(sources)

**Description:** Return four lists of filenames containing
C, C++, Fortran, and Fortran 90 module sources,
respectively.

### Function: _get_headers(directory_list)

### Function: _get_directories(list_of_sources)

### Function: _commandline_dep_string(cc_args, extra_postargs, pp_opts)

**Description:** Return commandline representation used to determine if a file needs
to be recompiled

### Function: get_dependencies(sources)

### Function: is_local_src_dir(directory)

**Description:** Return true if directory is local directory.
    

### Function: general_source_files(top_path)

### Function: general_source_directories_files(top_path)

**Description:** Return a directory name relative to top_path and
files contained.

### Function: get_ext_source_files(ext)

### Function: get_script_files(scripts)

### Function: get_lib_source_files(lib)

### Function: get_shared_lib_extension(is_python_ext)

**Description:** Return the correct file extension for shared libraries.

Parameters
----------
is_python_ext : bool, optional
    Whether the shared library is a Python extension.  Default is False.

Returns
-------
so_ext : str
    The shared library extension.

Notes
-----
For Python shared libs, `so_ext` will typically be '.so' on Linux and OS X,
and '.pyd' on Windows.  For Python >= 3.2 `so_ext` has a tag prepended on
POSIX systems according to PEP 3149.

### Function: get_data_files(data)

### Function: dot_join()

### Function: get_frame(level)

**Description:** Return frame object from call stack with given level.
    

## Class: Configuration

### Function: get_cmd(cmdname, _cache)

### Function: get_numpy_include_dirs()

### Function: get_npy_pkg_dir()

**Description:** Return the path where to find the npy-pkg-config directory.

If the NPY_PKG_CONFIG_PATH environment variable is set, the value of that
is returned.  Otherwise, a path inside the location of the numpy module is
returned.

The NPY_PKG_CONFIG_PATH can be useful when cross-compiling, maintaining
customized npy-pkg-config .ini files for the cross-compilation
environment, and using them when cross-compiling.

### Function: get_pkg_info(pkgname, dirs)

**Description:** Return library info for the given package.

Parameters
----------
pkgname : str
    Name of the package (should match the name of the .ini file, without
    the extension, e.g. foo for the file foo.ini).
dirs : sequence, optional
    If given, should be a sequence of additional directories where to look
    for npy-pkg-config files. Those directories are searched prior to the
    NumPy directory.

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
Configuration.add_npy_pkg_config, Configuration.add_installed_library,
get_info

### Function: get_info(pkgname, dirs)

**Description:** Return an info dict for a given C library.

The info dict contains the necessary options to use the C library.

Parameters
----------
pkgname : str
    Name of the package (should match the name of the .ini file, without
    the extension, e.g. foo for the file foo.ini).
dirs : sequence, optional
    If given, should be a sequence of additional directories where to look
    for npy-pkg-config files. Those directories are searched prior to the
    NumPy directory.

Returns
-------
info : dict
    The dictionary with build information.

Raises
------
PkgNotFound
    If the package is not found.

See Also
--------
Configuration.add_npy_pkg_config, Configuration.add_installed_library,
get_pkg_info

Examples
--------
To get the necessary information for the npymath library from NumPy:

>>> npymath_info = np.distutils.misc_util.get_info('npymath')
>>> npymath_info                                    #doctest: +SKIP
{'define_macros': [], 'libraries': ['npymath'], 'library_dirs':
['.../numpy/_core/lib'], 'include_dirs': ['.../numpy/_core/include']}

This info dict can then be used as input to a `Configuration` instance::

  config.add_extension('foo', sources=['foo.c'], extra_info=npymath_info)

### Function: is_bootstrapping()

### Function: default_config_dict(name, parent_name, local_path)

**Description:** Return a configuration dictionary for usage in
configuration() function defined in file setup_<name>.py.

### Function: dict_append(d)

### Function: appendpath(prefix, path)

### Function: generate_config_py(target)

**Description:** Generate config.py file containing system_info information
used during building the package.

Usage:
    config['py_modules'].append((packagename, '__config__',generate_config_py))

### Function: msvc_version(compiler)

**Description:** Return version major and minor of compiler instance if it is
MSVC, raise an exception otherwise.

### Function: get_build_architecture()

### Function: sanitize_cxx_flags(cxxflags)

**Description:** Some flags are valid for C but not C++. Prune them.

### Function: exec_mod_from_location(modname, modfile)

**Description:** Use importlib machinery to import a module `modname` from the file
`modfile`. Depending on the `spec.loader`, the module may not be
registered in sys.modules.

### Function: __init__(self, name, build_info, target_dir)

### Function: colour_text(s, fg, bg, bold)

### Function: colour_text(s, fg, bg)

### Function: __init__(self, package_name, parent_name, top_path, package_path, caller_level, setup_name)

**Description:** Construct configuration instance of a package.

package_name -- name of the package
                Ex.: 'distutils'
parent_name  -- name of the parent package
                Ex.: 'numpy'
top_path     -- directory of the toplevel package
                Ex.: the directory where the numpy package source sits
package_path -- directory of package. Will be computed by magic from the
                directory of the caller module if not specified
                Ex.: the directory where numpy.distutils is
caller_level -- frame level to caller namespace, internal parameter.

### Function: todict(self)

**Description:** Return a dictionary compatible with the keyword arguments of distutils
setup function.

Examples
--------
>>> setup(**config.todict())                           #doctest: +SKIP

### Function: info(self, message)

### Function: warn(self, message)

### Function: set_options(self)

**Description:** Configure Configuration instance.

The following options are available:
 - ignore_setup_xxx_py
 - assume_default_configuration
 - delegate_options_to_subpackages
 - quiet

### Function: get_distribution(self)

**Description:** Return the distutils distribution object for self.

### Function: _wildcard_get_subpackage(self, subpackage_name, parent_name, caller_level)

### Function: _get_configuration_from_setup_py(self, setup_py, subpackage_name, subpackage_path, parent_name, caller_level)

### Function: get_subpackage(self, subpackage_name, subpackage_path, parent_name, caller_level)

**Description:** Return list of subpackage configurations.

Parameters
----------
subpackage_name : str or None
    Name of the subpackage to get the configuration. '*' in
    subpackage_name is handled as a wildcard.
subpackage_path : str
    If None, then the path is assumed to be the local path plus the
    subpackage_name. If a setup.py file is not found in the
    subpackage_path, then a default configuration is used.
parent_name : str
    Parent name.

### Function: add_subpackage(self, subpackage_name, subpackage_path, standalone)

**Description:** Add a sub-package to the current Configuration instance.

This is useful in a setup.py script for adding sub-packages to a
package.

Parameters
----------
subpackage_name : str
    name of the subpackage
subpackage_path : str
    if given, the subpackage path such as the subpackage is in
    subpackage_path / subpackage_name. If None,the subpackage is
    assumed to be located in the local path / subpackage_name.
standalone : bool

### Function: add_data_dir(self, data_path)

**Description:** Recursively add files under data_path to data_files list.

Recursively add files under data_path to the list of data_files to be
installed (and distributed). The data_path can be either a relative
path-name, or an absolute path-name, or a 2-tuple where the first
argument shows where in the install directory the data directory
should be installed to.

Parameters
----------
data_path : seq or str
    Argument can be either

        * 2-sequence (<datadir suffix>, <path to data directory>)
        * path to data directory where python datadir suffix defaults
          to package dir.

Notes
-----
Rules for installation paths::

    foo/bar -> (foo/bar, foo/bar) -> parent/foo/bar
    (gun, foo/bar) -> parent/gun
    foo/* -> (foo/a, foo/a), (foo/b, foo/b) -> parent/foo/a, parent/foo/b
    (gun, foo/*) -> (gun, foo/a), (gun, foo/b) -> gun
    (gun/*, foo/*) -> parent/gun/a, parent/gun/b
    /foo/bar -> (bar, /foo/bar) -> parent/bar
    (gun, /foo/bar) -> parent/gun
    (fun/*/gun/*, sun/foo/bar) -> parent/fun/foo/gun/bar

Examples
--------
For example suppose the source directory contains fun/foo.dat and
fun/bar/car.dat:

>>> self.add_data_dir('fun')                       #doctest: +SKIP
>>> self.add_data_dir(('sun', 'fun'))              #doctest: +SKIP
>>> self.add_data_dir(('gun', '/full/path/to/fun'))#doctest: +SKIP

Will install data-files to the locations::

    <package install directory>/
      fun/
        foo.dat
        bar/
          car.dat
      sun/
        foo.dat
        bar/
          car.dat
      gun/
        foo.dat
        car.dat

### Function: _optimize_data_files(self)

### Function: add_data_files(self)

**Description:** Add data files to configuration data_files.

Parameters
----------
files : sequence
    Argument(s) can be either

        * 2-sequence (<datadir prefix>,<path to data file(s)>)
        * paths to data files where python datadir prefix defaults
          to package dir.

Notes
-----
The form of each element of the files sequence is very flexible
allowing many combinations of where to get the files from the package
and where they should ultimately be installed on the system. The most
basic usage is for an element of the files argument sequence to be a
simple filename. This will cause that file from the local path to be
installed to the installation path of the self.name package (package
path). The file argument can also be a relative path in which case the
entire relative path will be installed into the package directory.
Finally, the file can be an absolute path name in which case the file
will be found at the absolute path name but installed to the package
path.

This basic behavior can be augmented by passing a 2-tuple in as the
file argument. The first element of the tuple should specify the
relative path (under the package install directory) where the
remaining sequence of files should be installed to (it has nothing to
do with the file-names in the source distribution). The second element
of the tuple is the sequence of files that should be installed. The
files in this sequence can be filenames, relative paths, or absolute
paths. For absolute paths the file will be installed in the top-level
package installation directory (regardless of the first argument).
Filenames and relative path names will be installed in the package
install directory under the path name given as the first element of
the tuple.

Rules for installation paths:

  #. file.txt -> (., file.txt)-> parent/file.txt
  #. foo/file.txt -> (foo, foo/file.txt) -> parent/foo/file.txt
  #. /foo/bar/file.txt -> (., /foo/bar/file.txt) -> parent/file.txt
  #. ``*``.txt -> parent/a.txt, parent/b.txt
  #. foo/``*``.txt`` -> parent/foo/a.txt, parent/foo/b.txt
  #. ``*/*.txt`` -> (``*``, ``*``/``*``.txt) -> parent/c/a.txt, parent/d/b.txt
  #. (sun, file.txt) -> parent/sun/file.txt
  #. (sun, bar/file.txt) -> parent/sun/file.txt
  #. (sun, /foo/bar/file.txt) -> parent/sun/file.txt
  #. (sun, ``*``.txt) -> parent/sun/a.txt, parent/sun/b.txt
  #. (sun, bar/``*``.txt) -> parent/sun/a.txt, parent/sun/b.txt
  #. (sun/``*``, ``*``/``*``.txt) -> parent/sun/c/a.txt, parent/d/b.txt

An additional feature is that the path to a data-file can actually be
a function that takes no arguments and returns the actual path(s) to
the data-files. This is useful when the data files are generated while
building the package.

Examples
--------
Add files to the list of data_files to be included with the package.

    >>> self.add_data_files('foo.dat',
    ...     ('fun', ['gun.dat', 'nun/pun.dat', '/tmp/sun.dat']),
    ...     'bar/cat.dat',
    ...     '/full/path/to/can.dat')                   #doctest: +SKIP

will install these data files to::

    <package install directory>/
     foo.dat
     fun/
       gun.dat
       nun/
         pun.dat
     sun.dat
     bar/
       car.dat
     can.dat

where <package install directory> is the package (or sub-package)
directory such as '/usr/lib/python2.4/site-packages/mypackage' ('C:
\Python2.4 \Lib \site-packages \mypackage') or
'/usr/lib/python2.4/site- packages/mypackage/mysubpackage' ('C:
\Python2.4 \Lib \site-packages \mypackage \mysubpackage').

### Function: add_define_macros(self, macros)

**Description:** Add define macros to configuration

Add the given sequence of macro name and value duples to the beginning
of the define_macros list This list will be visible to all extension
modules of the current package.

### Function: add_include_dirs(self)

**Description:** Add paths to configuration include directories.

Add the given sequence of paths to the beginning of the include_dirs
list. This list will be visible to all extension modules of the
current package.

### Function: add_headers(self)

**Description:** Add installable headers to configuration.

Add the given sequence of files to the beginning of the headers list.
By default, headers will be installed under <python-
include>/<self.name.replace('.','/')>/ directory. If an item of files
is a tuple, then its first argument specifies the actual installation
location relative to the <python-include> path.

Parameters
----------
files : str or seq
    Argument(s) can be either:

        * 2-sequence (<includedir suffix>,<path to header file(s)>)
        * path(s) to header file(s) where python includedir suffix will
          default to package name.

### Function: paths(self)

**Description:** Apply glob to paths and prepend local_path if needed.

Applies glob.glob(...) to each path in the sequence (if needed) and
prepends the local_path if needed. Because this is called on all
source lists, this allows wildcard characters to be specified in lists
of sources for extension modules and libraries and scripts and allows
path-names be relative to the source directory.

### Function: _fix_paths_dict(self, kw)

### Function: add_extension(self, name, sources)

**Description:** Add extension to configuration.

Create and add an Extension instance to the ext_modules list. This
method also takes the following optional keyword arguments that are
passed on to the Extension constructor.

Parameters
----------
name : str
    name of the extension
sources : seq
    list of the sources. The list of sources may contain functions
    (called source generators) which must take an extension instance
    and a build directory as inputs and return a source file or list of
    source files or None. If None is returned then no sources are
    generated. If the Extension instance has no sources after
    processing all source generators, then no extension module is
    built.
include_dirs :
define_macros :
undef_macros :
library_dirs :
libraries :
runtime_library_dirs :
extra_objects :
extra_compile_args :
extra_link_args :
extra_f77_compile_args :
extra_f90_compile_args :
export_symbols :
swig_opts :
depends :
    The depends list contains paths to files or directories that the
    sources of the extension module depend on. If any path in the
    depends list is newer than the extension module, then the module
    will be rebuilt.
language :
f2py_options :
module_dirs :
extra_info : dict or list
    dict or list of dict of keywords to be appended to keywords.

Notes
-----
The self.paths(...) method is applied to all lists that may contain
paths.

### Function: add_library(self, name, sources)

**Description:** Add library to configuration.

Parameters
----------
name : str
    Name of the extension.
sources : sequence
    List of the sources. The list of sources may contain functions
    (called source generators) which must take an extension instance
    and a build directory as inputs and return a source file or list of
    source files or None. If None is returned then no sources are
    generated. If the Extension instance has no sources after
    processing all source generators, then no extension module is
    built.
build_info : dict, optional
    The following keys are allowed:

        * depends
        * macros
        * include_dirs
        * extra_compiler_args
        * extra_f77_compile_args
        * extra_f90_compile_args
        * f2py_options
        * language

### Function: _add_library(self, name, sources, install_dir, build_info)

**Description:** Common implementation for add_library and add_installed_library. Do
not use directly

### Function: add_installed_library(self, name, sources, install_dir, build_info)

**Description:** Similar to add_library, but the specified library is installed.

Most C libraries used with ``distutils`` are only used to build python
extensions, but libraries built through this method will be installed
so that they can be reused by third-party packages.

Parameters
----------
name : str
    Name of the installed library.
sources : sequence
    List of the library's source files. See `add_library` for details.
install_dir : str
    Path to install the library, relative to the current sub-package.
build_info : dict, optional
    The following keys are allowed:

        * depends
        * macros
        * include_dirs
        * extra_compiler_args
        * extra_f77_compile_args
        * extra_f90_compile_args
        * f2py_options
        * language

Returns
-------
None

See Also
--------
add_library, add_npy_pkg_config, get_info

Notes
-----
The best way to encode the options required to link against the specified
C libraries is to use a "libname.ini" file, and use `get_info` to
retrieve the required options (see `add_npy_pkg_config` for more
information).

### Function: add_npy_pkg_config(self, template, install_dir, subst_dict)

**Description:** Generate and install a npy-pkg config file from a template.

The config file generated from `template` is installed in the
given install directory, using `subst_dict` for variable substitution.

Parameters
----------
template : str
    The path of the template, relatively to the current package path.
install_dir : str
    Where to install the npy-pkg config file, relatively to the current
    package path.
subst_dict : dict, optional
    If given, any string of the form ``@key@`` will be replaced by
    ``subst_dict[key]`` in the template file when installed. The install
    prefix is always available through the variable ``@prefix@``, since the
    install prefix is not easy to get reliably from setup.py.

See also
--------
add_installed_library, get_info

Notes
-----
This works for both standard installs and in-place builds, i.e. the
``@prefix@`` refer to the source directory for in-place builds.

Examples
--------
::

    config.add_npy_pkg_config('foo.ini.in', 'lib', {'foo': bar})

Assuming the foo.ini.in file has the following content::

    [meta]
    Name=@foo@
    Version=1.0
    Description=dummy description

    [default]
    Cflags=-I@prefix@/include
    Libs=

The generated file will have the following content::

    [meta]
    Name=bar
    Version=1.0
    Description=dummy description

    [default]
    Cflags=-Iprefix_dir/include
    Libs=

and will be installed as foo.ini in the 'lib' subpath.

When cross-compiling with numpy distutils, it might be necessary to
use modified npy-pkg-config files.  Using the default/generated files
will link with the host libraries (i.e. libnpymath.a).  For
cross-compilation you of-course need to link with target libraries,
while using the host Python installation.

You can copy out the numpy/_core/lib/npy-pkg-config directory, add a
pkgdir value to the .ini files and set NPY_PKG_CONFIG_PATH environment
variable to point to the directory with the modified npy-pkg-config
files.

Example npymath.ini modified for cross-compilation::

    [meta]
    Name=npymath
    Description=Portable, core math library implementing C99 standard
    Version=0.1

    [variables]
    pkgname=numpy._core
    pkgdir=/build/arm-linux-gnueabi/sysroot/usr/lib/python3.7/site-packages/numpy/_core
    prefix=${pkgdir}
    libdir=${prefix}/lib
    includedir=${prefix}/include

    [default]
    Libs=-L${libdir} -lnpymath
    Cflags=-I${includedir}
    Requires=mlib

    [msvc]
    Libs=/LIBPATH:${libdir} npymath.lib
    Cflags=/INCLUDE:${includedir}
    Requires=mlib

### Function: add_scripts(self)

**Description:** Add scripts to configuration.

Add the sequence of files to the beginning of the scripts list.
Scripts will be installed under the <prefix>/bin/ directory.

### Function: dict_append(self)

### Function: __str__(self)

### Function: get_config_cmd(self)

**Description:** Returns the numpy.distutils config command instance.

### Function: get_build_temp_dir(self)

**Description:** Return a path to a temporary directory where temporary files should be
placed.

### Function: have_f77c(self)

**Description:** Check for availability of Fortran 77 compiler.

Use it inside source generating function to ensure that
setup distribution instance has been initialized.

Notes
-----
True if a Fortran 77 compiler is available (because a simple Fortran 77
code was able to be compiled successfully).

### Function: have_f90c(self)

**Description:** Check for availability of Fortran 90 compiler.

Use it inside source generating function to ensure that
setup distribution instance has been initialized.

Notes
-----
True if a Fortran 90 compiler is available (because a simple Fortran
90 code was able to be compiled successfully)

### Function: append_to(self, extlib)

**Description:** Append libraries, include_dirs to extension or library item.
        

### Function: _get_svn_revision(self, path)

**Description:** Return path's SVN revision number.
        

### Function: _get_hg_revision(self, path)

**Description:** Return path's Mercurial revision number.
        

### Function: get_version(self, version_file, version_variable)

**Description:** Try to get version string of a package.

Return a version string of the current package or None if the version
information could not be detected.

Notes
-----
This method scans files named
__version__.py, <packagename>_version.py, version.py, and
__svn_version__.py for string variables version, __version__, and
<packagename>_version, until a version number is found.

### Function: make_svn_version_py(self, delete)

**Description:** Appends a data function to the data_files list that will generate
__svn_version__.py file to the current package directory.

Generate package __svn_version__.py file from SVN revision number,
it will be removed after python exits but will be available
when sdist, etc commands are executed.

Notes
-----
If __svn_version__.py existed before, nothing is done.

This is
intended for working with source directories that are in an SVN
repository.

### Function: make_hg_version_py(self, delete)

**Description:** Appends a data function to the data_files list that will generate
__hg_version__.py file to the current package directory.

Generate package __hg_version__.py file from Mercurial revision,
it will be removed after python exits but will be available
when sdist, etc commands are executed.

Notes
-----
If __hg_version__.py existed before, nothing is done.

This is intended for working with source directories that are
in an Mercurial repository.

### Function: make_config_py(self, name)

**Description:** Generate package __config__.py file containing system_info
information used during building the package.

This file is installed to the
package installation directory.

### Function: get_info(self)

**Description:** Get resources information.

Return information (from system_info.get_info) for all of the names in
the argument list in a single dictionary.

### Function: generate_svn_version_py()

### Function: generate_hg_version_py()

### Function: rm_file(f, p)

### Function: rm_file(f, p)
