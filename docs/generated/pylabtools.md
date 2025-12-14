## AI Summary

A file named pylabtools.py.


### Function: __getattr__(name)

### Function: getfigs()

**Description:** Get a list of matplotlib figures by figure numbers.

If no arguments are given, all available figures are returned.  If the
argument list contains references to invalid figures, a warning is printed
but the function continues pasting further figures.

Parameters
----------
figs : tuple
    A tuple of ints giving the figure numbers of the figures to return.

### Function: figsize(sizex, sizey)

**Description:** Set the default figure size to be [sizex, sizey].

This is just an easy to remember, convenience wrapper that sets::

  matplotlib.rcParams['figure.figsize'] = [sizex, sizey]

### Function: print_figure(fig, fmt, bbox_inches, base64)

**Description:** Print a figure to an image, and return the resulting file data

Returned data will be bytes unless ``fmt='svg'``,
in which case it will be unicode.

Any keyword args are passed to fig.canvas.print_figure,
such as ``quality`` or ``bbox_inches``.

If `base64` is True, return base64-encoded str instead of raw bytes
for binary-encoded image formats

.. versionadded:: 7.29
    base64 argument

### Function: retina_figure(fig, base64)

**Description:** format a figure as a pixel-doubled (retina) PNG

If `base64` is True, return base64-encoded str instead of raw bytes
for binary-encoded image formats

.. versionadded:: 7.29
    base64 argument

### Function: mpl_runner(safe_execfile)

**Description:** Factory to return a matplotlib-enabled runner for %run.

Parameters
----------
safe_execfile : function
    This must be a function with the same interface as the
    :meth:`safe_execfile` method of IPython.

Returns
-------
A function suitable for use as the ``runner`` argument of the %run magic
function.

### Function: _reshow_nbagg_figure(fig)

**Description:** reshow an nbagg figure

### Function: select_figure_formats(shell, formats)

**Description:** Select figure formats for the inline backend.

Parameters
----------
shell : InteractiveShell
    The main IPython instance.
formats : str or set
    One or a set of figure formats to enable: 'png', 'retina', 'jpeg', 'svg', 'pdf'.
**kwargs : any
    Extra keyword arguments to be passed to fig.canvas.print_figure.

### Function: find_gui_and_backend(gui, gui_select)

**Description:** Given a gui string return the gui and mpl backend.

Parameters
----------
gui : str
    Can be one of ('tk','gtk','wx','qt','qt4','inline','agg').
gui_select : str
    Can be one of ('tk','gtk','wx','qt','qt4','inline').
    This is any gui already selected by the shell.

Returns
-------
A tuple of (gui, backend) where backend is one of ('TkAgg','GTKAgg',
'WXAgg','Qt4Agg','module://matplotlib_inline.backend_inline','agg').

### Function: activate_matplotlib(backend)

**Description:** Activate the given backend and set interactive to True.

### Function: import_pylab(user_ns, import_all)

**Description:** Populate the namespace with pylab-related values.

Imports matplotlib, pylab, numpy, and everything from pylab and numpy.

Also imports a few names from IPython (figsize, display, getfigs)

### Function: configure_inline_support(shell, backend)

**Description:** .. deprecated:: 7.23

    use `matplotlib_inline.backend_inline.configure_inline_support()`

Configure an IPython shell object for matplotlib use.

Parameters
----------
shell : InteractiveShell instance
backend : matplotlib backend

### Function: _matplotlib_manages_backends()

**Description:** Return True if Matplotlib manages backends, False otherwise.

If it returns True, the caller can be sure that
matplotlib.backends.registry.backend_registry is available along with
member functions resolve_gui_or_backend, resolve_backend, list_all, and
list_gui_frameworks.

This function can be removed as it will always return True when Python
3.12, the latest version supported by Matplotlib < 3.9, reaches
end-of-life in late 2028.

### Function: _list_matplotlib_backends_and_gui_loops()

**Description:** Return list of all Matplotlib backends and GUI event loops.

This is the list returned by
    %matplotlib --list

### Function: _convert_gui_to_matplotlib(gui)

### Function: _convert_gui_from_matplotlib(gui)

### Function: mpl_execfile(fname)

**Description:** matplotlib-aware wrapper around safe_execfile.

Its interface is identical to that of the :func:`execfile` builtin.

This is ultimately a call to execfile(), but wrapped in safeties to
properly handle interactive rendering.
