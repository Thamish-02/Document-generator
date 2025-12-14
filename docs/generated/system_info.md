## AI Summary

A file named system_info.py.


### Function: customized_ccompiler()

### Function: _c_string_literal(s)

**Description:** Convert a python string into a literal suitable for inclusion into C code

### Function: libpaths(paths, bits)

**Description:** Return a list of library paths valid on 32 or 64 bit systems.

Inputs:
  paths : sequence
    A sequence of strings (typically paths)
  bits : int
    An integer, the only valid values are 32 or 64.  A ValueError exception
  is raised otherwise.

Examples:

Consider a list of directories
>>> paths = ['/usr/X11R6/lib','/usr/X11/lib','/usr/lib']

For a 32-bit platform, this is already valid:
>>> np.distutils.system_info.libpaths(paths,32)
['/usr/X11R6/lib', '/usr/X11/lib', '/usr/lib']

On 64 bits, we prepend the '64' postfix
>>> np.distutils.system_info.libpaths(paths,64)
['/usr/X11R6/lib64', '/usr/X11R6/lib', '/usr/X11/lib64', '/usr/X11/lib',
'/usr/lib64', '/usr/lib']

### Function: get_standard_file(fname)

**Description:** Returns a list of files named 'fname' from
1) System-wide directory (directory-location of this module)
2) Users HOME directory (os.environ['HOME'])
3) Local directory

### Function: _parse_env_order(base_order, env)

**Description:** Parse an environment variable `env` by splitting with "," and only returning elements from `base_order`

This method will sequence the environment variable and check for their
individual elements in `base_order`.

The items in the environment variable may be negated via '^item' or '!itema,itemb'.
It must start with ^/! to negate all options.

Raises
------
ValueError: for mixed negated and non-negated orders or multiple negated orders

Parameters
----------
base_order : list of str
   the base list of orders
env : str
   the environment variable to be parsed, if none is found, `base_order` is returned

Returns
-------
allow_order : list of str
    allowed orders in lower-case
unknown_order : list of str
    for values not overlapping with `base_order`

### Function: get_info(name, notfound_action)

**Description:** notfound_action:
  0 - do nothing
  1 - display warning message
  2 - raise error

## Class: NotFoundError

**Description:** Some third-party program or library is not found.

## Class: AliasedOptionError

**Description:** Aliases entries in config files should not be existing.
In section '{section}' we found multiple appearances of options {options}.

## Class: AtlasNotFoundError

**Description:** Atlas (http://github.com/math-atlas/math-atlas) libraries not found.
Directories to search for the libraries can be specified in the
numpy/distutils/site.cfg file (section [atlas]) or by setting
the ATLAS environment variable.

## Class: FlameNotFoundError

**Description:** FLAME (http://www.cs.utexas.edu/~flame/web/) libraries not found.
Directories to search for the libraries can be specified in the
numpy/distutils/site.cfg file (section [flame]).

## Class: LapackNotFoundError

**Description:** Lapack (http://www.netlib.org/lapack/) libraries not found.
Directories to search for the libraries can be specified in the
numpy/distutils/site.cfg file (section [lapack]) or by setting
the LAPACK environment variable.

## Class: LapackSrcNotFoundError

**Description:** Lapack (http://www.netlib.org/lapack/) sources not found.
Directories to search for the sources can be specified in the
numpy/distutils/site.cfg file (section [lapack_src]) or by setting
the LAPACK_SRC environment variable.

## Class: LapackILP64NotFoundError

**Description:** 64-bit Lapack libraries not found.
Known libraries in numpy/distutils/site.cfg file are:
openblas64_, openblas_ilp64

## Class: BlasOptNotFoundError

**Description:** Optimized (vendor) Blas libraries are not found.
Falls back to netlib Blas library which has worse performance.
A better performance should be easily gained by switching
Blas library.

## Class: BlasNotFoundError

**Description:** Blas (http://www.netlib.org/blas/) libraries not found.
Directories to search for the libraries can be specified in the
numpy/distutils/site.cfg file (section [blas]) or by setting
the BLAS environment variable.

## Class: BlasILP64NotFoundError

**Description:** 64-bit Blas libraries not found.
Known libraries in numpy/distutils/site.cfg file are:
openblas64_, openblas_ilp64

## Class: BlasSrcNotFoundError

**Description:** Blas (http://www.netlib.org/blas/) sources not found.
Directories to search for the sources can be specified in the
numpy/distutils/site.cfg file (section [blas_src]) or by setting
the BLAS_SRC environment variable.

## Class: FFTWNotFoundError

**Description:** FFTW (http://www.fftw.org/) libraries not found.
Directories to search for the libraries can be specified in the
numpy/distutils/site.cfg file (section [fftw]) or by setting
the FFTW environment variable.

## Class: DJBFFTNotFoundError

**Description:** DJBFFT (https://cr.yp.to/djbfft.html) libraries not found.
Directories to search for the libraries can be specified in the
numpy/distutils/site.cfg file (section [djbfft]) or by setting
the DJBFFT environment variable.

## Class: NumericNotFoundError

**Description:** Numeric (https://www.numpy.org/) module not found.
Get it from above location, install it, and retry setup.py.

## Class: X11NotFoundError

**Description:** X11 libraries not found.

## Class: UmfpackNotFoundError

**Description:** UMFPACK sparse solver (https://www.cise.ufl.edu/research/sparse/umfpack/)
not found. Directories to search for the libraries can be specified in the
numpy/distutils/site.cfg file (section [umfpack]) or by setting
the UMFPACK environment variable.

## Class: system_info

**Description:** get_info() is the only public method. Don't use others.
    

## Class: fft_opt_info

## Class: fftw_info

## Class: fftw2_info

## Class: fftw3_info

## Class: fftw3_armpl_info

## Class: dfftw_info

## Class: sfftw_info

## Class: fftw_threads_info

## Class: dfftw_threads_info

## Class: sfftw_threads_info

## Class: djbfft_info

## Class: mkl_info

## Class: lapack_mkl_info

## Class: blas_mkl_info

## Class: ssl2_info

## Class: lapack_ssl2_info

## Class: blas_ssl2_info

## Class: armpl_info

## Class: lapack_armpl_info

## Class: blas_armpl_info

## Class: atlas_info

## Class: atlas_blas_info

## Class: atlas_threads_info

## Class: atlas_blas_threads_info

## Class: lapack_atlas_info

## Class: lapack_atlas_threads_info

## Class: atlas_3_10_info

## Class: atlas_3_10_blas_info

## Class: atlas_3_10_threads_info

## Class: atlas_3_10_blas_threads_info

## Class: lapack_atlas_3_10_info

## Class: lapack_atlas_3_10_threads_info

## Class: lapack_info

## Class: lapack_src_info

### Function: get_atlas_version()

## Class: lapack_opt_info

## Class: _ilp64_opt_info_mixin

## Class: lapack_ilp64_opt_info

## Class: lapack_ilp64_plain_opt_info

## Class: lapack64__opt_info

## Class: blas_opt_info

## Class: blas_ilp64_opt_info

## Class: blas_ilp64_plain_opt_info

## Class: blas64__opt_info

## Class: cblas_info

## Class: blas_info

## Class: openblas_info

## Class: openblas_lapack_info

## Class: openblas_clapack_info

## Class: openblas_ilp64_info

## Class: openblas_ilp64_lapack_info

## Class: openblas64__info

## Class: openblas64__lapack_info

## Class: blis_info

## Class: flame_info

**Description:** Usage of libflame for LAPACK operations

This requires libflame to be compiled with lapack wrappers:

./configure --enable-lapack2flame ...

Be aware that libflame 5.1.0 has some missing names in the shared library, so
if you have problems, try the static flame library.

## Class: accelerate_info

## Class: accelerate_lapack_info

## Class: blas_src_info

## Class: x11_info

## Class: _numpy_info

## Class: numarray_info

## Class: Numeric_info

## Class: numpy_info

## Class: numerix_info

## Class: f2py_info

## Class: boost_python_info

## Class: agg2_info

## Class: _pkg_config_info

## Class: wx_info

## Class: gdk_pixbuf_xlib_2_info

## Class: gdk_pixbuf_2_info

## Class: gdk_x11_2_info

## Class: gdk_2_info

## Class: gdk_info

## Class: gtkp_x11_2_info

## Class: gtkp_2_info

## Class: xft_info

## Class: freetype2_info

## Class: amd_info

## Class: umfpack_info

### Function: combine_paths()

**Description:** Return a list of existing paths composed by all combinations of
items from arguments.

### Function: dict_append(d)

### Function: parseCmdLine(argv)

### Function: show_all(argv)

### Function: add_system_root(library_root)

**Description:** Add a package manager root to the include directories

### Function: __init__(self, default_lib_dirs, default_include_dirs)

### Function: parse_config_files(self)

### Function: calc_libraries_info(self)

### Function: set_info(self)

### Function: get_option_single(self)

**Description:** Ensure that only one of `options` are found in the section

Parameters
----------
*options : list of str
   a list of options to be found in the section (``self.section``)

Returns
-------
str :
    the option that is uniquely found in the section

Raises
------
AliasedOptionError :
    in case more than one of the options are found

### Function: has_info(self)

### Function: calc_extra_info(self)

**Description:** Updates the information in the current information with
respect to these flags:
  extra_compile_args
  extra_link_args

### Function: get_info(self, notfound_action)

**Description:** Return a dictionary with items that are compatible
with numpy.distutils.setup keyword arguments.

### Function: get_paths(self, section, key)

### Function: get_lib_dirs(self, key)

### Function: get_runtime_lib_dirs(self, key)

### Function: get_include_dirs(self, key)

### Function: get_src_dirs(self, key)

### Function: get_libs(self, key, default)

### Function: get_libraries(self, key)

### Function: library_extensions(self)

### Function: check_libs(self, lib_dirs, libs, opt_libs)

**Description:** If static or shared libraries are available then return
their info dictionary.

Checks for all libraries as shared libraries first, then
static (or vice versa if self.search_static_first is True).

### Function: check_libs2(self, lib_dirs, libs, opt_libs)

**Description:** If static or shared libraries are available then return
their info dictionary.

Checks each library for shared or static.

### Function: _find_lib(self, lib_dir, lib, exts)

### Function: _find_libs(self, lib_dirs, libs, exts)

### Function: _check_libs(self, lib_dirs, libs, opt_libs, exts)

**Description:** Find mandatory and optional libs in expected paths.

Missing optional libraries are silently forgotten.

### Function: combine_paths(self)

**Description:** Return a list of existing paths composed by all combinations
of items from the arguments.

### Function: calc_info(self)

### Function: calc_ver_info(self, ver_param)

**Description:** Returns True on successful version detection, else False

### Function: calc_info(self)

### Function: get_paths(self, section, key)

### Function: calc_info(self)

### Function: get_mkl_rootdir(self)

### Function: __init__(self)

### Function: calc_info(self)

### Function: get_tcsds_rootdir(self)

### Function: __init__(self)

### Function: calc_info(self)

### Function: calc_info(self)

### Function: get_paths(self, section, key)

### Function: calc_info(self)

### Function: calc_info(self)

### Function: calc_info(self)

### Function: calc_info(self)

### Function: get_paths(self, section, key)

### Function: calc_info(self)

### Function: _calc_info_armpl(self)

### Function: _calc_info_mkl(self)

### Function: _calc_info_ssl2(self)

### Function: _calc_info_openblas(self)

### Function: _calc_info_flame(self)

### Function: _calc_info_atlas(self)

### Function: _calc_info_accelerate(self)

### Function: _get_info_blas(self)

### Function: _get_info_lapack(self)

### Function: _calc_info_lapack(self)

### Function: _calc_info_from_envvar(self)

### Function: _calc_info(self, name)

### Function: calc_info(self)

### Function: _check_info(self, info)

### Function: _calc_info(self, name)

### Function: _calc_info_armpl(self)

### Function: _calc_info_mkl(self)

### Function: _calc_info_ssl2(self)

### Function: _calc_info_blis(self)

### Function: _calc_info_openblas(self)

### Function: _calc_info_atlas(self)

### Function: _calc_info_accelerate(self)

### Function: _calc_info_blas(self)

### Function: _calc_info_from_envvar(self)

### Function: _calc_info(self, name)

### Function: calc_info(self)

### Function: _calc_info(self, name)

### Function: calc_info(self)

### Function: get_cblas_libs(self, info)

**Description:** Check whether we can link with CBLAS interface

This method will search through several combinations of libraries
to check whether CBLAS is present:

1. Libraries in ``info['libraries']``, as is
2. As 1. but also explicitly adding ``'cblas'`` as a library
3. As 1. but also explicitly adding ``'blas'`` as a library
4. Check only library ``'cblas'``
5. Check only library ``'blas'``

Parameters
----------
info : dict
   system information dictionary for compilation and linking

Returns
-------
libraries : list of str or None
    a list of libraries that enables the use of CBLAS interface.
    Returns None if not found or a compilation error occurs.

    Since 1.17 returns a list.

### Function: symbol_prefix(self)

### Function: symbol_suffix(self)

### Function: _calc_info(self)

### Function: calc_info(self)

### Function: check_msvc_gfortran_libs(self, library_dirs, libraries)

### Function: check_symbols(self, info)

### Function: _calc_info(self)

### Function: _calc_info(self)

### Function: calc_info(self)

### Function: check_embedded_lapack(self, info)

**Description:** libflame does not necessarily have a wrapper for fortran LAPACK, we need to check 

### Function: calc_info(self)

### Function: calc_info(self)

### Function: _calc_info(self)

### Function: get_paths(self, section, key)

### Function: calc_info(self)

### Function: __init__(self)

### Function: calc_info(self)

### Function: __init__(self)

### Function: calc_info(self)

### Function: calc_info(self)

### Function: calc_info(self)

### Function: get_paths(self, section, key)

### Function: calc_info(self)

### Function: get_paths(self, section, key)

### Function: calc_info(self)

### Function: get_config_exe(self)

### Function: get_config_output(self, config_exe, option)

### Function: calc_info(self)

### Function: calc_info(self)

### Function: calc_info(self)
