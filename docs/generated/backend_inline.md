## AI Summary

A file named backend_inline.py.


### Function: new_figure_manager(num)

**Description:** Return a new figure manager for a new figure instance.

This function is part of the API expected by Matplotlib backends.

### Function: new_figure_manager_given_figure(num, figure)

**Description:** Return a new figure manager for a given figure instance.

This function is part of the API expected by Matplotlib backends.

### Function: show(close, block)

**Description:** Show all figures as SVG/PNG payloads sent to the IPython clients.

Parameters
----------
close : bool, optional
    If true, a ``plt.close('all')`` call is automatically issued after
    sending all the figures. If this is set, the figures will entirely
    removed from the internal list of figures.
block : Not used.
    The `block` parameter is a Matplotlib experimental parameter.
    We accept it in the function signature for compatibility with other
    backends.

### Function: flush_figures()

**Description:** Send all figures that changed

This is meant to be called automatically and will call show() if, during
prior code execution, there had been any calls to draw_if_interactive.

This function is meant to be used as a post_execute callback in IPython,
so user-caused errors are handled with showtraceback() instead of being
allowed to raise.  If this function is not called from within IPython,
then these exceptions will raise.

### Function: configure_inline_support(shell, backend)

**Description:** Configure an IPython shell object for matplotlib use.

Parameters
----------
shell : InteractiveShell instance

backend : matplotlib backend

### Function: _enable_matplotlib_integration()

**Description:** Enable extra IPython matplotlib integration when we are loaded as the matplotlib backend.

### Function: _fetch_figure_metadata(fig)

**Description:** Get some metadata to help with displaying a figure.

### Function: _is_light(color)

**Description:** Determines if a color (or each of a sequence of colors) is light (as
opposed to dark). Based on ITU BT.601 luminance formula (see
https://stackoverflow.com/a/596241).

### Function: _is_transparent(color)

**Description:** Determine transparency from alpha.

### Function: set_matplotlib_formats()

**Description:** Select figure formats for the inline backend. Optionally pass quality for JPEG.

For example, this enables PNG and JPEG output with a JPEG quality of 90%::

    In [1]: set_matplotlib_formats('png', 'jpeg',
                                   pil_kwargs={'quality': 90})

To set this in your notebook by `%config` magic::

    In [1]: %config InlineBackend.figure_formats = {'png', 'jpeg'}
            %config InlineBackend.print_figure_kwargs = \
                                    {'pil_kwargs': {'quality' : 90}}

To set this in your config files use the following::

    c.InlineBackend.figure_formats = {'png', 'jpeg'}
    c.InlineBackend.print_figure_kwargs.update({
                                    'pil_kwargs': {'quality' : 90}
                                })

Parameters
----------
*formats : strs
    One or more figure formats to enable: 'png', 'retina', 'jpeg', 'svg', 'pdf'.
**kwargs
    Keyword args will be relayed to ``figure.canvas.print_figure``.

In addition, see the docstrings of `plt.savefig()`,
`matplotlib.figure.Figure.savefig()`, `PIL.Image.Image.save()` and
:ref:`Pillow Image file formats <handbook/image-file-formats>`.

### Function: set_matplotlib_close(close)

**Description:** Set whether the inline backend closes all figures automatically or not.

By default, the inline backend used in the IPython Notebook will close all
matplotlib figures automatically after each cell is run. This means that
plots in different cells won't interfere. Sometimes, you may want to make
a plot in one cell and then refine it in later cells. This can be accomplished
by::

    In [1]: set_matplotlib_close(False)

To set this in your config files use the following::

    c.InlineBackend.close_figures = False

Parameters
----------
close : bool
    Should all matplotlib figures be automatically closed after each cell is
    run?

### Function: configure_once()
