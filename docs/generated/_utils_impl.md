## AI Summary

A file named _utils_impl.py.


### Function: show_runtime()

**Description:** Print information about various resources in the system
including available intrinsic support and BLAS/LAPACK library
in use

.. versionadded:: 1.24.0

See Also
--------
show_config : Show libraries in the system on which NumPy was built.

Notes
-----
1. Information is derived with the help of `threadpoolctl <https://pypi.org/project/threadpoolctl/>`_
   library if available.
2. SIMD related information is derived from ``__cpu_features__``,
   ``__cpu_baseline__`` and ``__cpu_dispatch__``

### Function: get_include()

**Description:** Return the directory that contains the NumPy \*.h header files.

Extension modules that need to compile against NumPy may need to use this
function to locate the appropriate include directory.

Notes
-----
When using ``setuptools``, for example in ``setup.py``::

    import numpy as np
    ...
    Extension('extension_name', ...
              include_dirs=[np.get_include()])
    ...

Note that a CLI tool ``numpy-config`` was introduced in NumPy 2.0, using
that is likely preferred for build systems other than ``setuptools``::

    $ numpy-config --cflags
    -I/path/to/site-packages/numpy/_core/include

    # Or rely on pkg-config:
    $ export PKG_CONFIG_PATH=$(numpy-config --pkgconfigdir)
    $ pkg-config --cflags
    -I/path/to/site-packages/numpy/_core/include

Examples
--------
>>> np.get_include()
'.../site-packages/numpy/core/include'  # may vary

## Class: _Deprecate

**Description:** Decorator class to deprecate old functions.

Refer to `deprecate` for details.

See Also
--------
deprecate

### Function: _get_indent(lines)

**Description:** Determines the leading whitespace that could be removed from all the lines.

### Function: deprecate()

**Description:** Issues a DeprecationWarning, adds warning to `old_name`'s
docstring, rebinds ``old_name.__name__`` and returns the new
function object.

This function may also be used as a decorator.

.. deprecated:: 2.0
    Use `~warnings.warn` with :exc:`DeprecationWarning` instead.

Parameters
----------
func : function
    The function to be deprecated.
old_name : str, optional
    The name of the function to be deprecated. Default is None, in
    which case the name of `func` is used.
new_name : str, optional
    The new name for the function. Default is None, in which case the
    deprecation message is that `old_name` is deprecated. If given, the
    deprecation message is that `old_name` is deprecated and `new_name`
    should be used instead.
message : str, optional
    Additional explanation of the deprecation.  Displayed in the
    docstring after the warning.

Returns
-------
old_func : function
    The deprecated function.

Examples
--------
Note that ``olduint`` returns a value after printing Deprecation
Warning:

>>> olduint = np.lib.utils.deprecate(np.uint)
DeprecationWarning: `uint64` is deprecated! # may vary
>>> olduint(6)
6

### Function: deprecate_with_doc(msg)

**Description:** Deprecates a function and includes the deprecation in its docstring.

.. deprecated:: 2.0
    Use `~warnings.warn` with :exc:`DeprecationWarning` instead.

This function is used as a decorator. It returns an object that can be
used to issue a DeprecationWarning, by passing the to-be decorated
function as argument, this adds warning to the to-be decorated function's
docstring and returns the new function object.

See Also
--------
deprecate : Decorate a function such that it issues a
            :exc:`DeprecationWarning`

Parameters
----------
msg : str
    Additional explanation of the deprecation. Displayed in the
    docstring after the warning.

Returns
-------
obj : object

### Function: _split_line(name, arguments, width)

### Function: _makenamedict(module)

### Function: _info(obj, output)

**Description:** Provide information about ndarray obj.

Parameters
----------
obj : ndarray
    Must be ndarray, not checked.
output
    Where printed output goes.

Notes
-----
Copied over from the numarray module prior to its removal.
Adapted somewhat as only numpy is an option now.

Called by info.

### Function: info(object, maxwidth, output, toplevel)

**Description:** Get help information for an array, function, class, or module.

Parameters
----------
object : object or str, optional
    Input object or name to get information about. If `object` is
    an `ndarray` instance, information about the array is printed.
    If `object` is a numpy object, its docstring is given. If it is
    a string, available modules are searched for matching objects.
    If None, information about `info` itself is returned.
maxwidth : int, optional
    Printing width.
output : file like object, optional
    File like object that the output is written to, default is
    ``None``, in which case ``sys.stdout`` will be used.
    The object has to be opened in 'w' or 'a' mode.
toplevel : str, optional
    Start search at this level.

Notes
-----
When used interactively with an object, ``np.info(obj)`` is equivalent
to ``help(obj)`` on the Python prompt or ``obj?`` on the IPython
prompt.

Examples
--------
>>> np.info(np.polyval) # doctest: +SKIP
   polyval(p, x)
     Evaluate the polynomial p at x.
     ...

When using a string for `object` it is possible to get multiple results.

>>> np.info('fft') # doctest: +SKIP
     *** Found in numpy ***
Core FFT routines
...
     *** Found in numpy.fft ***
 fft(a, n=None, axis=-1)
...
     *** Repeat reference found in numpy.fft.fftpack ***
     *** Total of 3 references found. ***

When the argument is an array, information about the array is printed.

>>> a = np.array([[1 + 2j, 3, -4], [-5j, 6, 0]], dtype=np.complex64)
>>> np.info(a)
class:  ndarray
shape:  (2, 3)
strides:  (24, 8)
itemsize:  8
aligned:  True
contiguous:  True
fortran:  False
data pointer: 0x562b6e0d2860  # may vary
byteorder:  little
byteswap:  False
type: complex64

### Function: safe_eval(source)

**Description:** Protected string evaluation.

.. deprecated:: 2.0
    Use `ast.literal_eval` instead.

Evaluate a string containing a Python literal expression without
allowing the execution of arbitrary non-literal code.

.. warning::

    This function is identical to :py:meth:`ast.literal_eval` and
    has the same security implications.  It may not always be safe
    to evaluate large input strings.

Parameters
----------
source : str
    The string to evaluate.

Returns
-------
obj : object
   The result of evaluating `source`.

Raises
------
SyntaxError
    If the code has invalid Python syntax, or if it contains
    non-literal code.

Examples
--------
>>> np.safe_eval('1')
1
>>> np.safe_eval('[1, 2, 3]')
[1, 2, 3]
>>> np.safe_eval('{"foo": ("bar", 10.0)}')
{'foo': ('bar', 10.0)}

>>> np.safe_eval('import os')
Traceback (most recent call last):
  ...
SyntaxError: invalid syntax

>>> np.safe_eval('open("/home/user/.ssh/id_dsa").read()')
Traceback (most recent call last):
  ...
ValueError: malformed node or string: <_ast.Call object at 0x...>

### Function: _median_nancheck(data, result, axis)

**Description:** Utility function to check median result from data for NaN values at the end
and return NaN in that case. Input result can also be a MaskedArray.

Parameters
----------
data : array
    Sorted input data to median function
result : Array or MaskedArray
    Result of median function.
axis : int
    Axis along which the median was computed.

Returns
-------
result : scalar or ndarray
    Median or NaN in axes which contained NaN in the input.  If the input
    was an array, NaN will be inserted in-place.  If a scalar, either the
    input itself or a scalar NaN.

### Function: _opt_info()

**Description:** Returns a string containing the CPU features supported
by the current build.

The format of the string can be explained as follows:
    - Dispatched features supported by the running machine end with `*`.
    - Dispatched features not supported by the running machine
      end with `?`.
    - Remaining features represent the baseline.

Returns:
    str: A formatted string indicating the supported CPU features.

### Function: drop_metadata()

**Description:** Returns the dtype unchanged if it contained no metadata or a copy of the
dtype if it (or any of its structure dtypes) contained metadata.

This utility is used by `np.save` and `np.savez` to drop metadata before
saving.

.. note::

    Due to its limitation this function may move to a more appropriate
    home or change in the future and is considered semi-public API only.

.. warning::

    This function does not preserve more strange things like record dtypes
    and user dtypes may simply return the wrong thing.  If you need to be
    sure about the latter, check the result with:
    ``np.can_cast(new_dtype, dtype, casting="no")``.

### Function: __init__(self, old_name, new_name, message)

### Function: __call__(self, func)

**Description:** Decorator call.  Refer to ``decorate``.

### Function: newfunc()
