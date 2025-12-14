## AI Summary

A file named pyplot.py.


### Function: _copy_docstring_and_deprecators(method, func)

### Function: _copy_docstring_and_deprecators(method, func)

### Function: _copy_docstring_and_deprecators(method, func)

### Function: _add_pyplot_note(func, wrapped_func)

**Description:** Add a note to the docstring of *func* that it is a pyplot wrapper.

The note is added to the "Notes" section of the docstring. If that does
not exist, a "Notes" section is created. In numpydoc, the "Notes"
section is the third last possible section, only potentially followed by
"References" and "Examples".

### Function: _draw_all_if_interactive()

### Function: install_repl_displayhook()

**Description:** Connect to the display hook of the current shell.

The display hook gets called when the read-evaluate-print-loop (REPL) of
the shell has finished the execution of a command. We use this callback
to be able to automatically update a figure in interactive mode.

This works both with IPython and with vanilla python shells.

### Function: uninstall_repl_displayhook()

**Description:** Disconnect from the display hook of the current shell.

### Function: set_loglevel()

### Function: findobj(o, match, include_self)

### Function: _get_backend_mod()

**Description:** Ensure that a backend is selected and return it.

This is currently private, but may be made public in the future.

### Function: switch_backend(newbackend)

**Description:** Set the pyplot backend.

Switching to an interactive backend is possible only if no event loop for
another interactive backend has started.  Switching to and from
non-interactive backends is always possible.

If the new backend is different than the current backend then all open
Figures will be closed via ``plt.close('all')``.

Parameters
----------
newbackend : str
    The case-insensitive name of the backend to use.

### Function: _warn_if_gui_out_of_main_thread()

### Function: new_figure_manager()

**Description:** Create a new figure manager instance.

### Function: draw_if_interactive()

**Description:** Redraw the current figure if in interactive mode.

.. warning::

    End users will typically not have to call this function because the
    the interactive mode takes care of this.

### Function: show()

**Description:** Display all open figures.

Parameters
----------
block : bool, optional
    Whether to wait for all figures to be closed before returning.

    If `True` block and run the GUI main loop until all figure windows
    are closed.

    If `False` ensure that all figure windows are displayed and return
    immediately.  In this case, you are responsible for ensuring
    that the event loop is running to have responsive figures.

    Defaults to True in non-interactive mode and to False in interactive
    mode (see `.pyplot.isinteractive`).

See Also
--------
ion : Enable interactive mode, which shows / updates the figure after
      every plotting command, so that calling ``show()`` is not necessary.
ioff : Disable interactive mode.
savefig : Save the figure to an image file instead of showing it on screen.

Notes
-----
**Saving figures to file and showing a window at the same time**

If you want an image file as well as a user interface window, use
`.pyplot.savefig` before `.pyplot.show`. At the end of (a blocking)
``show()`` the figure is closed and thus unregistered from pyplot. Calling
`.pyplot.savefig` afterwards would save a new and thus empty figure. This
limitation of command order does not apply if the show is non-blocking or
if you keep a reference to the figure and use `.Figure.savefig`.

**Auto-show in jupyter notebooks**

The jupyter backends (activated via ``%matplotlib inline``,
``%matplotlib notebook``, or ``%matplotlib widget``), call ``show()`` at
the end of every cell by default. Thus, you usually don't have to call it
explicitly there.

### Function: isinteractive()

**Description:** Return whether plots are updated after every plotting command.

The interactive mode is mainly useful if you build plots from the command
line and want to see the effect of each command while you are building the
figure.

In interactive mode:

- newly created figures will be shown immediately;
- figures will automatically redraw on change;
- `.pyplot.show` will not block by default.

In non-interactive mode:

- newly created figures and changes to figures will not be reflected until
  explicitly asked to be;
- `.pyplot.show` will block by default.

See Also
--------
ion : Enable interactive mode.
ioff : Disable interactive mode.
show : Show all figures (and maybe block).
pause : Show all figures, and block for a time.

### Function: ioff()

**Description:** Disable interactive mode.

See `.pyplot.isinteractive` for more details.

See Also
--------
ion : Enable interactive mode.
isinteractive : Whether interactive mode is enabled.
show : Show all figures (and maybe block).
pause : Show all figures, and block for a time.

Notes
-----
For a temporary change, this can be used as a context manager::

    # if interactive mode is on
    # then figures will be shown on creation
    plt.ion()
    # This figure will be shown immediately
    fig = plt.figure()

    with plt.ioff():
        # interactive mode will be off
        # figures will not automatically be shown
        fig2 = plt.figure()
        # ...

To enable optional usage as a context manager, this function returns a
context manager object, which is not intended to be stored or
accessed by the user.

### Function: ion()

**Description:** Enable interactive mode.

See `.pyplot.isinteractive` for more details.

See Also
--------
ioff : Disable interactive mode.
isinteractive : Whether interactive mode is enabled.
show : Show all figures (and maybe block).
pause : Show all figures, and block for a time.

Notes
-----
For a temporary change, this can be used as a context manager::

    # if interactive mode is off
    # then figures will not be shown on creation
    plt.ioff()
    # This figure will not be shown immediately
    fig = plt.figure()

    with plt.ion():
        # interactive mode will be on
        # figures will automatically be shown
        fig2 = plt.figure()
        # ...

To enable optional usage as a context manager, this function returns a
context manager object, which is not intended to be stored or
accessed by the user.

### Function: pause(interval)

**Description:** Run the GUI event loop for *interval* seconds.

If there is an active figure, it will be updated and displayed before the
pause, and the GUI event loop (if any) will run during the pause.

This can be used for crude animation.  For more complex animation use
:mod:`matplotlib.animation`.

If there is no active figure, sleep for *interval* seconds instead.

See Also
--------
matplotlib.animation : Proper animations
show : Show all figures and optional block until all figures are closed.

### Function: rc(group)

### Function: rc_context(rc, fname)

### Function: rcdefaults()

### Function: getp(obj)

### Function: get(obj)

### Function: setp(obj)

### Function: xkcd(scale, length, randomness)

**Description:** Turn on `xkcd <https://xkcd.com/>`_ sketch-style drawing mode.

This will only have an effect on things drawn after this function is called.

For best results, install the `xkcd script <https://github.com/ipython/xkcd-font/>`_
font; xkcd fonts are not packaged with Matplotlib.

Parameters
----------
scale : float, optional
    The amplitude of the wiggle perpendicular to the source line.
length : float, optional
    The length of the wiggle along the line.
randomness : float, optional
    The scale factor by which the length is shrunken or expanded.

Notes
-----
This function works by a number of rcParams, so it will probably
override others you have set before.

If you want the effects of this function to be temporary, it can
be used as a context manager, for example::

    with plt.xkcd():
        # This figure will be in XKCD-style
        fig1 = plt.figure()
        # ...

    # This figure will be in regular style
    fig2 = plt.figure()

### Function: figure(num, figsize, dpi)

**Description:** Create a new figure, or activate an existing figure.

Parameters
----------
num : int or str or `.Figure` or `.SubFigure`, optional
    A unique identifier for the figure.

    If a figure with that identifier already exists, this figure is made
    active and returned. An integer refers to the ``Figure.number``
    attribute, a string refers to the figure label.

    If there is no figure with the identifier or *num* is not given, a new
    figure is created, made active and returned.  If *num* is an int, it
    will be used for the ``Figure.number`` attribute, otherwise, an
    auto-generated integer value is used (starting at 1 and incremented
    for each new figure). If *num* is a string, the figure label and the
    window title is set to this value.  If num is a ``SubFigure``, its
    parent ``Figure`` is activated.

figsize : (float, float), default: :rc:`figure.figsize`
    Width, height in inches.

dpi : float, default: :rc:`figure.dpi`
    The resolution of the figure in dots-per-inch.

facecolor : :mpltype:`color`, default: :rc:`figure.facecolor`
    The background color.

edgecolor : :mpltype:`color`, default: :rc:`figure.edgecolor`
    The border color.

frameon : bool, default: True
    If False, suppress drawing the figure frame.

FigureClass : subclass of `~matplotlib.figure.Figure`
    If set, an instance of this subclass will be created, rather than a
    plain `.Figure`.

clear : bool, default: False
    If True and the figure already exists, then it is cleared.

layout : {'constrained', 'compressed', 'tight', 'none', `.LayoutEngine`, None}, default: None
    The layout mechanism for positioning of plot elements to avoid
    overlapping Axes decorations (labels, ticks, etc). Note that layout
    managers can measurably slow down figure display.

    - 'constrained': The constrained layout solver adjusts Axes sizes
      to avoid overlapping Axes decorations.  Can handle complex plot
      layouts and colorbars, and is thus recommended.

      See :ref:`constrainedlayout_guide`
      for examples.

    - 'compressed': uses the same algorithm as 'constrained', but
      removes extra space between fixed-aspect-ratio Axes.  Best for
      simple grids of Axes.

    - 'tight': Use the tight layout mechanism. This is a relatively
      simple algorithm that adjusts the subplot parameters so that
      decorations do not overlap. See `.Figure.set_tight_layout` for
      further details.

    - 'none': Do not use a layout engine.

    - A `.LayoutEngine` instance. Builtin layout classes are
      `.ConstrainedLayoutEngine` and `.TightLayoutEngine`, more easily
      accessible by 'constrained' and 'tight'.  Passing an instance
      allows third parties to provide their own layout engine.

    If not given, fall back to using the parameters *tight_layout* and
    *constrained_layout*, including their config defaults
    :rc:`figure.autolayout` and :rc:`figure.constrained_layout.use`.

**kwargs
    Additional keyword arguments are passed to the `.Figure` constructor.

Returns
-------
`~matplotlib.figure.Figure`

Notes
-----
A newly created figure is passed to the `~.FigureCanvasBase.new_manager`
method or the `new_figure_manager` function provided by the current
backend, which install a canvas and a manager on the figure.

Once this is done, :rc:`figure.hooks` are called, one at a time, on the
figure; these hooks allow arbitrary customization of the figure (e.g.,
attaching callbacks) or of associated elements (e.g., modifying the
toolbar).  See :doc:`/gallery/user_interfaces/mplcvd` for an example of
toolbar customization.

If you are creating many figures, make sure you explicitly call
`.pyplot.close` on the figures you are not using, because this will
enable pyplot to properly clean up the memory.

`~matplotlib.rcParams` defines the default values, which can be modified
in the matplotlibrc file.

### Function: _auto_draw_if_interactive(fig, val)

**Description:** An internal helper function for making sure that auto-redrawing
works as intended in the plain python repl.

Parameters
----------
fig : Figure
    A figure object which is assumed to be associated with a canvas

### Function: gcf()

**Description:** Get the current figure.

If there is currently no figure on the pyplot figure stack, a new one is
created using `~.pyplot.figure()`.  (To test whether there is currently a
figure on the pyplot figure stack, check whether `~.pyplot.get_fignums()`
is empty.)

### Function: fignum_exists(num)

**Description:** Return whether the figure with the given id exists.

Parameters
----------
num : int or str
    A figure identifier.

Returns
-------
bool
    Whether or not a figure with id *num* exists.

### Function: get_fignums()

**Description:** Return a list of existing figure numbers.

### Function: get_figlabels()

**Description:** Return a list of existing figure labels.

### Function: get_current_fig_manager()

**Description:** Return the figure manager of the current figure.

The figure manager is a container for the actual backend-depended window
that displays the figure on screen.

If no current figure exists, a new one is created, and its figure
manager is returned.

Returns
-------
`.FigureManagerBase` or backend-dependent subclass thereof

### Function: connect(s, func)

### Function: disconnect(cid)

### Function: close(fig)

**Description:** Close a figure window, and unregister it from pyplot.

Parameters
----------
fig : None or int or str or `.Figure`
    The figure to close. There are a number of ways to specify this:

    - *None*: the current figure
    - `.Figure`: the given `.Figure` instance
    - ``int``: a figure number
    - ``str``: a figure name
    - 'all': all figures

Notes
-----
pyplot maintains a reference to figures created with `figure()`. When
work on the figure is completed, it should be closed, i.e. deregistered
from pyplot, to free its memory (see also :rc:figure.max_open_warning).
Closing a figure window created by `show()` automatically deregisters the
figure. For all other use cases, most prominently `savefig()` without
`show()`, the figure must be deregistered explicitly using `close()`.

### Function: clf()

**Description:** Clear the current figure.

### Function: draw()

**Description:** Redraw the current figure.

This is used to update a figure that has been altered, but not
automatically re-drawn.  If interactive mode is on (via `.ion()`), this
should be only rarely needed, but there may be ways to modify the state of
a figure without marking it as "stale".  Please report these cases as bugs.

This is equivalent to calling ``fig.canvas.draw_idle()``, where ``fig`` is
the current figure.

See Also
--------
.FigureCanvasBase.draw_idle
.FigureCanvasBase.draw

### Function: savefig()

### Function: figlegend()

### Function: axes(arg)

**Description:** Add an Axes to the current figure and make it the current Axes.

Call signatures::

    plt.axes()
    plt.axes(rect, projection=None, polar=False, **kwargs)
    plt.axes(ax)

Parameters
----------
arg : None or 4-tuple
    The exact behavior of this function depends on the type:

    - *None*: A new full window Axes is added using
      ``subplot(**kwargs)``.
    - 4-tuple of floats *rect* = ``(left, bottom, width, height)``.
      A new Axes is added with dimensions *rect* in normalized
      (0, 1) units using `~.Figure.add_axes` on the current figure.

projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
    The projection type of the `~.axes.Axes`. *str* is the name of
    a custom projection, see `~matplotlib.projections`. The default
    None results in a 'rectilinear' projection.

polar : bool, default: False
    If True, equivalent to projection='polar'.

sharex, sharey : `~matplotlib.axes.Axes`, optional
    Share the x or y `~matplotlib.axis` with sharex and/or sharey.
    The axis will have the same limits, ticks, and scale as the axis
    of the shared Axes.

label : str
    A label for the returned Axes.

Returns
-------
`~.axes.Axes`, or a subclass of `~.axes.Axes`
    The returned Axes class depends on the projection used. It is
    `~.axes.Axes` if rectilinear projection is used and
    `.projections.polar.PolarAxes` if polar projection is used.

Other Parameters
----------------
**kwargs
    This method also takes the keyword arguments for
    the returned Axes class. The keyword arguments for the
    rectilinear Axes class `~.axes.Axes` can be found in
    the following table but there might also be other keyword
    arguments if another projection is used, see the actual Axes
    class.

    %(Axes:kwdoc)s

See Also
--------
.Figure.add_axes
.pyplot.subplot
.Figure.add_subplot
.Figure.subplots
.pyplot.subplots

Examples
--------
::

    # Creating a new full window Axes
    plt.axes()

    # Creating a new Axes with specified dimensions and a grey background
    plt.axes((left, bottom, width, height), facecolor='grey')

### Function: delaxes(ax)

**Description:** Remove an `~.axes.Axes` (defaulting to the current Axes) from its figure.

### Function: sca(ax)

**Description:** Set the current Axes to *ax* and the current Figure to the parent of *ax*.

### Function: cla()

**Description:** Clear the current Axes.

### Function: subplot()

**Description:** Add an Axes to the current figure or retrieve an existing Axes.

This is a wrapper of `.Figure.add_subplot` which provides additional
behavior when working with the implicit API (see the notes section).

Call signatures::

   subplot(nrows, ncols, index, **kwargs)
   subplot(pos, **kwargs)
   subplot(**kwargs)
   subplot(ax)

Parameters
----------
*args : int, (int, int, *index*), or `.SubplotSpec`, default: (1, 1, 1)
    The position of the subplot described by one of

    - Three integers (*nrows*, *ncols*, *index*). The subplot will take the
      *index* position on a grid with *nrows* rows and *ncols* columns.
      *index* starts at 1 in the upper left corner and increases to the
      right. *index* can also be a two-tuple specifying the (*first*,
      *last*) indices (1-based, and including *last*) of the subplot, e.g.,
      ``fig.add_subplot(3, 1, (1, 2))`` makes a subplot that spans the
      upper 2/3 of the figure.
    - A 3-digit integer. The digits are interpreted as if given separately
      as three single-digit integers, i.e. ``fig.add_subplot(235)`` is the
      same as ``fig.add_subplot(2, 3, 5)``. Note that this can only be used
      if there are no more than 9 subplots.
    - A `.SubplotSpec`.

projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
    The projection type of the subplot (`~.axes.Axes`). *str* is the name
    of a custom projection, see `~matplotlib.projections`. The default
    None results in a 'rectilinear' projection.

polar : bool, default: False
    If True, equivalent to projection='polar'.

sharex, sharey : `~matplotlib.axes.Axes`, optional
    Share the x or y `~matplotlib.axis` with sharex and/or sharey. The
    axis will have the same limits, ticks, and scale as the axis of the
    shared Axes.

label : str
    A label for the returned Axes.

Returns
-------
`~.axes.Axes`

    The Axes of the subplot. The returned Axes can actually be an instance
    of a subclass, such as `.projections.polar.PolarAxes` for polar
    projections.

Other Parameters
----------------
**kwargs
    This method also takes the keyword arguments for the returned Axes
    base class; except for the *figure* argument. The keyword arguments
    for the rectilinear base class `~.axes.Axes` can be found in
    the following table but there might also be other keyword
    arguments if another projection is used.

    %(Axes:kwdoc)s

Notes
-----
.. versionchanged:: 3.8
    In versions prior to 3.8, any preexisting Axes that overlap with the new Axes
    beyond sharing a boundary was deleted. Deletion does not happen in more
    recent versions anymore. Use `.Axes.remove` explicitly if needed.

If you do not want this behavior, use the `.Figure.add_subplot` method
or the `.pyplot.axes` function instead.

If no *kwargs* are passed and there exists an Axes in the location
specified by *args* then that Axes will be returned rather than a new
Axes being created.

If *kwargs* are passed and there exists an Axes in the location
specified by *args*, the projection type is the same, and the
*kwargs* match with the existing Axes, then the existing Axes is
returned.  Otherwise a new Axes is created with the specified
parameters.  We save a reference to the *kwargs* which we use
for this comparison.  If any of the values in *kwargs* are
mutable we will not detect the case where they are mutated.
In these cases we suggest using `.Figure.add_subplot` and the
explicit Axes API rather than the implicit pyplot API.

See Also
--------
.Figure.add_subplot
.pyplot.subplots
.pyplot.axes
.Figure.subplots

Examples
--------
::

    plt.subplot(221)

    # equivalent but more general
    ax1 = plt.subplot(2, 2, 1)

    # add a subplot with no frame
    ax2 = plt.subplot(222, frameon=False)

    # add a polar subplot
    plt.subplot(223, projection='polar')

    # add a red subplot that shares the x-axis with ax1
    plt.subplot(224, sharex=ax1, facecolor='red')

    # delete ax2 from the figure
    plt.delaxes(ax2)

    # add ax2 to the figure again
    plt.subplot(ax2)

    # make the first Axes "current" again
    plt.subplot(221)

### Function: subplots(nrows, ncols)

### Function: subplots(nrows, ncols)

### Function: subplots(nrows, ncols)

### Function: subplots(nrows, ncols)

**Description:** Create a figure and a set of subplots.

This utility wrapper makes it convenient to create common layouts of
subplots, including the enclosing figure object, in a single call.

Parameters
----------
nrows, ncols : int, default: 1
    Number of rows/columns of the subplot grid.

sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
    Controls sharing of properties among x (*sharex*) or y (*sharey*)
    axes:

    - True or 'all': x- or y-axis will be shared among all subplots.
    - False or 'none': each subplot x- or y-axis will be independent.
    - 'row': each subplot row will share an x- or y-axis.
    - 'col': each subplot column will share an x- or y-axis.

    When subplots have a shared x-axis along a column, only the x tick
    labels of the bottom subplot are created. Similarly, when subplots
    have a shared y-axis along a row, only the y tick labels of the first
    column subplot are created. To later turn other subplots' ticklabels
    on, use `~matplotlib.axes.Axes.tick_params`.

    When subplots have a shared axis that has units, calling
    `.Axis.set_units` will update each axis with the new units.

    Note that it is not possible to unshare axes.

squeeze : bool, default: True
    - If True, extra dimensions are squeezed out from the returned
      array of `~matplotlib.axes.Axes`:

      - if only one subplot is constructed (nrows=ncols=1), the
        resulting single Axes object is returned as a scalar.
      - for Nx1 or 1xM subplots, the returned object is a 1D numpy
        object array of Axes objects.
      - for NxM, subplots with N>1 and M>1 are returned as a 2D array.

    - If False, no squeezing at all is done: the returned Axes object is
      always a 2D array containing Axes instances, even if it ends up
      being 1x1.

width_ratios : array-like of length *ncols*, optional
    Defines the relative widths of the columns. Each column gets a
    relative width of ``width_ratios[i] / sum(width_ratios)``.
    If not given, all columns will have the same width.  Equivalent
    to ``gridspec_kw={'width_ratios': [...]}``.

height_ratios : array-like of length *nrows*, optional
    Defines the relative heights of the rows. Each row gets a
    relative height of ``height_ratios[i] / sum(height_ratios)``.
    If not given, all rows will have the same height. Convenience
    for ``gridspec_kw={'height_ratios': [...]}``.

subplot_kw : dict, optional
    Dict with keywords passed to the
    `~matplotlib.figure.Figure.add_subplot` call used to create each
    subplot.

gridspec_kw : dict, optional
    Dict with keywords passed to the `~matplotlib.gridspec.GridSpec`
    constructor used to create the grid the subplots are placed on.

**fig_kw
    All additional keyword arguments are passed to the
    `.pyplot.figure` call.

Returns
-------
fig : `.Figure`

ax : `~matplotlib.axes.Axes` or array of Axes
    *ax* can be either a single `~.axes.Axes` object, or an array of Axes
    objects if more than one subplot was created.  The dimensions of the
    resulting array can be controlled with the squeeze keyword, see above.

    Typical idioms for handling the return value are::

        # using the variable ax for single a Axes
        fig, ax = plt.subplots()

        # using the variable axs for multiple Axes
        fig, axs = plt.subplots(2, 2)

        # using tuple unpacking for multiple Axes
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    The names ``ax`` and pluralized ``axs`` are preferred over ``axes``
    because for the latter it's not clear if it refers to a single
    `~.axes.Axes` instance or a collection of these.

See Also
--------
.pyplot.figure
.pyplot.subplot
.pyplot.axes
.Figure.subplots
.Figure.add_subplot

Examples
--------
::

    # First create some toy data:
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # Create just a figure and only one subplot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Create four polar Axes and access them through the returned array
    fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
    axs[0, 0].plot(x, y)
    axs[1, 1].scatter(x, y)

    # Share a X axis with each column of subplots
    plt.subplots(2, 2, sharex='col')

    # Share a Y axis with each row of subplots
    plt.subplots(2, 2, sharey='row')

    # Share both X and Y axes with all subplots
    plt.subplots(2, 2, sharex='all', sharey='all')

    # Note that this is the same as
    plt.subplots(2, 2, sharex=True, sharey=True)

    # Create figure number 10 with a single subplot
    # and clears it if it already exists.
    fig, ax = plt.subplots(num=10, clear=True)

### Function: subplot_mosaic(mosaic)

### Function: subplot_mosaic(mosaic)

### Function: subplot_mosaic(mosaic)

### Function: subplot_mosaic(mosaic)

**Description:** Build a layout of Axes based on ASCII art or nested lists.

This is a helper function to build complex GridSpec layouts visually.

See :ref:`mosaic`
for an example and full API documentation

Parameters
----------
mosaic : list of list of {hashable or nested} or str

    A visual layout of how you want your Axes to be arranged
    labeled as strings.  For example ::

       x = [['A panel', 'A panel', 'edge'],
            ['C panel', '.',       'edge']]

    produces 4 Axes:

    - 'A panel' which is 1 row high and spans the first two columns
    - 'edge' which is 2 rows high and is on the right edge
    - 'C panel' which in 1 row and 1 column wide in the bottom left
    - a blank space 1 row and 1 column wide in the bottom center

    Any of the entries in the layout can be a list of lists
    of the same form to create nested layouts.

    If input is a str, then it must be of the form ::

      '''
      AAE
      C.E
      '''

    where each character is a column and each line is a row.
    This only allows only single character Axes labels and does
    not allow nesting but is very terse.

sharex, sharey : bool, default: False
    If True, the x-axis (*sharex*) or y-axis (*sharey*) will be shared
    among all subplots.  In that case, tick label visibility and axis units
    behave as for `subplots`.  If False, each subplot's x- or y-axis will
    be independent.

width_ratios : array-like of length *ncols*, optional
    Defines the relative widths of the columns. Each column gets a
    relative width of ``width_ratios[i] / sum(width_ratios)``.
    If not given, all columns will have the same width.  Convenience
    for ``gridspec_kw={'width_ratios': [...]}``.

height_ratios : array-like of length *nrows*, optional
    Defines the relative heights of the rows. Each row gets a
    relative height of ``height_ratios[i] / sum(height_ratios)``.
    If not given, all rows will have the same height. Convenience
    for ``gridspec_kw={'height_ratios': [...]}``.

empty_sentinel : object, optional
    Entry in the layout to mean "leave this space empty".  Defaults
    to ``'.'``. Note, if *layout* is a string, it is processed via
    `inspect.cleandoc` to remove leading white space, which may
    interfere with using white-space as the empty sentinel.

subplot_kw : dict, optional
    Dictionary with keywords passed to the `.Figure.add_subplot` call
    used to create each subplot.  These values may be overridden by
    values in *per_subplot_kw*.

per_subplot_kw : dict, optional
    A dictionary mapping the Axes identifiers or tuples of identifiers
    to a dictionary of keyword arguments to be passed to the
    `.Figure.add_subplot` call used to create each subplot.  The values
    in these dictionaries have precedence over the values in
    *subplot_kw*.

    If *mosaic* is a string, and thus all keys are single characters,
    it is possible to use a single string instead of a tuple as keys;
    i.e. ``"AB"`` is equivalent to ``("A", "B")``.

    .. versionadded:: 3.7

gridspec_kw : dict, optional
    Dictionary with keywords passed to the `.GridSpec` constructor used
    to create the grid the subplots are placed on.

**fig_kw
    All additional keyword arguments are passed to the
    `.pyplot.figure` call.

Returns
-------
fig : `.Figure`
   The new figure

dict[label, Axes]
   A dictionary mapping the labels to the Axes objects.  The order of
   the Axes is left-to-right and top-to-bottom of their position in the
   total layout.

### Function: subplot2grid(shape, loc, rowspan, colspan, fig)

**Description:** Create a subplot at a specific location inside a regular grid.

Parameters
----------
shape : (int, int)
    Number of rows and of columns of the grid in which to place axis.
loc : (int, int)
    Row number and column number of the axis location within the grid.
rowspan : int, default: 1
    Number of rows for the axis to span downwards.
colspan : int, default: 1
    Number of columns for the axis to span to the right.
fig : `.Figure`, optional
    Figure to place the subplot in. Defaults to the current figure.
**kwargs
    Additional keyword arguments are handed to `~.Figure.add_subplot`.

Returns
-------
`~.axes.Axes`

    The Axes of the subplot. The returned Axes can actually be an instance
    of a subclass, such as `.projections.polar.PolarAxes` for polar
    projections.

Notes
-----
The following call ::

    ax = subplot2grid((nrows, ncols), (row, col), rowspan, colspan)

is identical to ::

    fig = gcf()
    gs = fig.add_gridspec(nrows, ncols)
    ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])

### Function: twinx(ax)

**Description:** Make and return a second Axes that shares the *x*-axis.  The new Axes will
overlay *ax* (or the current Axes if *ax* is *None*), and its ticks will be
on the right.

Examples
--------
:doc:`/gallery/subplots_axes_and_figures/two_scales`

### Function: twiny(ax)

**Description:** Make and return a second Axes that shares the *y*-axis.  The new Axes will
overlay *ax* (or the current Axes if *ax* is *None*), and its ticks will be
on the top.

Examples
--------
:doc:`/gallery/subplots_axes_and_figures/two_scales`

### Function: subplot_tool(targetfig)

**Description:** Launch a subplot tool window for a figure.

Returns
-------
`matplotlib.widgets.SubplotTool`

### Function: box(on)

**Description:** Turn the Axes box on or off on the current Axes.

Parameters
----------
on : bool or None
    The new `~matplotlib.axes.Axes` box state. If ``None``, toggle
    the state.

See Also
--------
:meth:`matplotlib.axes.Axes.set_frame_on`
:meth:`matplotlib.axes.Axes.get_frame_on`

### Function: xlim()

**Description:** Get or set the x limits of the current Axes.

Call signatures::

    left, right = xlim()  # return the current xlim
    xlim((left, right))   # set the xlim to left, right
    xlim(left, right)     # set the xlim to left, right

If you do not specify args, you can pass *left* or *right* as kwargs,
i.e.::

    xlim(right=3)  # adjust the right leaving left unchanged
    xlim(left=1)  # adjust the left leaving right unchanged

Setting limits turns autoscaling off for the x-axis.

Returns
-------
left, right
    A tuple of the new x-axis limits.

Notes
-----
Calling this function with no arguments (e.g. ``xlim()``) is the pyplot
equivalent of calling `~.Axes.get_xlim` on the current Axes.
Calling this function with arguments is the pyplot equivalent of calling
`~.Axes.set_xlim` on the current Axes. All arguments are passed though.

### Function: ylim()

**Description:** Get or set the y-limits of the current Axes.

Call signatures::

    bottom, top = ylim()  # return the current ylim
    ylim((bottom, top))   # set the ylim to bottom, top
    ylim(bottom, top)     # set the ylim to bottom, top

If you do not specify args, you can alternatively pass *bottom* or
*top* as kwargs, i.e.::

    ylim(top=3)  # adjust the top leaving bottom unchanged
    ylim(bottom=1)  # adjust the bottom leaving top unchanged

Setting limits turns autoscaling off for the y-axis.

Returns
-------
bottom, top
    A tuple of the new y-axis limits.

Notes
-----
Calling this function with no arguments (e.g. ``ylim()``) is the pyplot
equivalent of calling `~.Axes.get_ylim` on the current Axes.
Calling this function with arguments is the pyplot equivalent of calling
`~.Axes.set_ylim` on the current Axes. All arguments are passed though.

### Function: xticks(ticks, labels)

**Description:** Get or set the current tick locations and labels of the x-axis.

Pass no arguments to return the current values without modifying them.

Parameters
----------
ticks : array-like, optional
    The list of xtick locations.  Passing an empty list removes all xticks.
labels : array-like, optional
    The labels to place at the given *ticks* locations.  This argument can
    only be passed if *ticks* is passed as well.
minor : bool, default: False
    If ``False``, get/set the major ticks/labels; if ``True``, the minor
    ticks/labels.
**kwargs
    `.Text` properties can be used to control the appearance of the labels.

    .. warning::

        This only sets the properties of the current ticks, which is
        only sufficient if you either pass *ticks*, resulting in a
        fixed list of ticks, or if the plot is static.

        Ticks are not guaranteed to be persistent. Various operations
        can create, delete and modify the Tick instances. There is an
        imminent risk that these settings can get lost if you work on
        the figure further (including also panning/zooming on a
        displayed figure).

        Use `~.pyplot.tick_params` instead if possible.


Returns
-------
locs
    The list of xtick locations.
labels
    The list of xlabel `.Text` objects.

Notes
-----
Calling this function with no arguments (e.g. ``xticks()``) is the pyplot
equivalent of calling `~.Axes.get_xticks` and `~.Axes.get_xticklabels` on
the current Axes.
Calling this function with arguments is the pyplot equivalent of calling
`~.Axes.set_xticks` and `~.Axes.set_xticklabels` on the current Axes.

Examples
--------
>>> locs, labels = xticks()  # Get the current locations and labels.
>>> xticks(np.arange(0, 1, step=0.2))  # Set label locations.
>>> xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
>>> xticks([0, 1, 2], ['January', 'February', 'March'],
...        rotation=20)  # Set text labels and properties.
>>> xticks([])  # Disable xticks.

### Function: yticks(ticks, labels)

**Description:** Get or set the current tick locations and labels of the y-axis.

Pass no arguments to return the current values without modifying them.

Parameters
----------
ticks : array-like, optional
    The list of ytick locations.  Passing an empty list removes all yticks.
labels : array-like, optional
    The labels to place at the given *ticks* locations.  This argument can
    only be passed if *ticks* is passed as well.
minor : bool, default: False
    If ``False``, get/set the major ticks/labels; if ``True``, the minor
    ticks/labels.
**kwargs
    `.Text` properties can be used to control the appearance of the labels.

    .. warning::

        This only sets the properties of the current ticks, which is
        only sufficient if you either pass *ticks*, resulting in a
        fixed list of ticks, or if the plot is static.

        Ticks are not guaranteed to be persistent. Various operations
        can create, delete and modify the Tick instances. There is an
        imminent risk that these settings can get lost if you work on
        the figure further (including also panning/zooming on a
        displayed figure).

        Use `~.pyplot.tick_params` instead if possible.

Returns
-------
locs
    The list of ytick locations.
labels
    The list of ylabel `.Text` objects.

Notes
-----
Calling this function with no arguments (e.g. ``yticks()``) is the pyplot
equivalent of calling `~.Axes.get_yticks` and `~.Axes.get_yticklabels` on
the current Axes.
Calling this function with arguments is the pyplot equivalent of calling
`~.Axes.set_yticks` and `~.Axes.set_yticklabels` on the current Axes.

Examples
--------
>>> locs, labels = yticks()  # Get the current locations and labels.
>>> yticks(np.arange(0, 1, step=0.2))  # Set label locations.
>>> yticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
>>> yticks([0, 1, 2], ['January', 'February', 'March'],
...        rotation=45)  # Set text labels and properties.
>>> yticks([])  # Disable yticks.

### Function: rgrids(radii, labels, angle, fmt)

**Description:** Get or set the radial gridlines on the current polar plot.

Call signatures::

 lines, labels = rgrids()
 lines, labels = rgrids(radii, labels=None, angle=22.5, fmt=None, **kwargs)

When called with no arguments, `.rgrids` simply returns the tuple
(*lines*, *labels*). When called with arguments, the labels will
appear at the specified radial distances and angle.

Parameters
----------
radii : tuple with floats
    The radii for the radial gridlines

labels : tuple with strings or None
    The labels to use at each radial gridline. The
    `matplotlib.ticker.ScalarFormatter` will be used if None.

angle : float
    The angular position of the radius labels in degrees.

fmt : str or None
    Format string used in `matplotlib.ticker.FormatStrFormatter`.
    For example '%f'.

Returns
-------
lines : list of `.lines.Line2D`
    The radial gridlines.

labels : list of `.text.Text`
    The tick labels.

Other Parameters
----------------
**kwargs
    *kwargs* are optional `.Text` properties for the labels.

See Also
--------
.pyplot.thetagrids
.projections.polar.PolarAxes.set_rgrids
.Axis.get_gridlines
.Axis.get_ticklabels

Examples
--------
::

  # set the locations of the radial gridlines
  lines, labels = rgrids( (0.25, 0.5, 1.0) )

  # set the locations and labels of the radial gridlines
  lines, labels = rgrids( (0.25, 0.5, 1.0), ('Tom', 'Dick', 'Harry' ))

### Function: thetagrids(angles, labels, fmt)

**Description:** Get or set the theta gridlines on the current polar plot.

Call signatures::

 lines, labels = thetagrids()
 lines, labels = thetagrids(angles, labels=None, fmt=None, **kwargs)

When called with no arguments, `.thetagrids` simply returns the tuple
(*lines*, *labels*). When called with arguments, the labels will
appear at the specified angles.

Parameters
----------
angles : tuple with floats, degrees
    The angles of the theta gridlines.

labels : tuple with strings or None
    The labels to use at each radial gridline. The
    `.projections.polar.ThetaFormatter` will be used if None.

fmt : str or None
    Format string used in `matplotlib.ticker.FormatStrFormatter`.
    For example '%f'. Note that the angle in radians will be used.

Returns
-------
lines : list of `.lines.Line2D`
    The theta gridlines.

labels : list of `.text.Text`
    The tick labels.

Other Parameters
----------------
**kwargs
    *kwargs* are optional `.Text` properties for the labels.

See Also
--------
.pyplot.rgrids
.projections.polar.PolarAxes.set_thetagrids
.Axis.get_gridlines
.Axis.get_ticklabels

Examples
--------
::

  # set the locations of the angular gridlines
  lines, labels = thetagrids(range(45, 360, 90))

  # set the locations and labels of the angular gridlines
  lines, labels = thetagrids(range(45, 360, 90), ('NE', 'NW', 'SW', 'SE'))

### Function: get_plot_commands()

**Description:** Get a sorted list of all of the plotting commands.

### Function: _get_pyplot_commands()

### Function: colorbar(mappable, cax, ax)

### Function: clim(vmin, vmax)

**Description:** Set the color limits of the current image.

If either *vmin* or *vmax* is None, the image min/max respectively
will be used for color scaling.

If you want to set the clim of multiple images, use
`~.ScalarMappable.set_clim` on every image, for example::

  for im in gca().get_images():
      im.set_clim(0, 0.5)

### Function: get_cmap(name, lut)

**Description:** Get a colormap instance, defaulting to rc values if *name* is None.

Parameters
----------
name : `~matplotlib.colors.Colormap` or str or None, default: None
    If a `.Colormap` instance, it will be returned. Otherwise, the name of
    a colormap known to Matplotlib, which will be resampled by *lut*. The
    default, None, means :rc:`image.cmap`.
lut : int or None, default: None
    If *name* is not already a Colormap instance and *lut* is not None, the
    colormap will be resampled to have *lut* entries in the lookup table.

Returns
-------
Colormap

### Function: set_cmap(cmap)

**Description:** Set the default colormap, and applies it to the current image if any.

Parameters
----------
cmap : `~matplotlib.colors.Colormap` or str
    A colormap instance or the name of a registered colormap.

See Also
--------
colormaps
get_cmap

### Function: imread(fname, format)

### Function: imsave(fname, arr)

### Function: matshow(A, fignum)

**Description:** Display a 2D array as a matrix in a new figure window.

The origin is set at the upper left hand corner.
The indexing is ``(row, column)`` so that the first index runs vertically
and the second index runs horizontally in the figure:

.. code-block:: none

    A[0, 0]   ⋯ A[0, M-1]
       ⋮             ⋮
    A[N-1, 0] ⋯ A[N-1, M-1]

The aspect ratio of the figure window is that of the array,
unless this would make an excessively short or narrow figure.

Tick labels for the xaxis are placed on top.

Parameters
----------
A : 2D array-like
    The matrix to be displayed.

fignum : None or int
    If *None*, create a new, appropriately sized figure window.

    If 0, use the current Axes (creating one if there is none, without ever
    adjusting the figure size).

    Otherwise, create a new Axes on the figure with the given number
    (creating it at the appropriate size if it does not exist, but not
    adjusting the figure size otherwise).  Note that this will be drawn on
    top of any preexisting Axes on the figure.

Returns
-------
`~matplotlib.image.AxesImage`

Other Parameters
----------------
**kwargs : `~matplotlib.axes.Axes.imshow` arguments

### Function: polar()

**Description:** Make a polar plot.

call signature::

  polar(theta, r, [fmt], **kwargs)

This is a convenience wrapper around `.pyplot.plot`. It ensures that the
current Axes is polar (or creates one if needed) and then passes all parameters
to ``.pyplot.plot``.

.. note::
    When making polar plots using the :ref:`pyplot API <pyplot_interface>`,
    ``polar()`` should typically be the first command because that makes sure
    a polar Axes is created. Using other commands such as ``plt.title()``
    before this can lead to the implicit creation of a rectangular Axes, in which
    case a subsequent ``polar()`` call will fail.

### Function: figimage(X, xo, yo, alpha, norm, cmap, vmin, vmax, origin, resize)

### Function: figtext(x, y, s, fontdict)

### Function: gca()

### Function: gci()

### Function: ginput(n, timeout, show_clicks, mouse_add, mouse_pop, mouse_stop)

### Function: subplots_adjust(left, bottom, right, top, wspace, hspace)

### Function: suptitle(t)

### Function: tight_layout()

### Function: waitforbuttonpress(timeout)

### Function: acorr(x)

### Function: angle_spectrum(x, Fs, Fc, window, pad_to, sides)

### Function: annotate(text, xy, xytext, xycoords, textcoords, arrowprops, annotation_clip)

### Function: arrow(x, y, dx, dy)

### Function: autoscale(enable, axis, tight)

### Function: axhline(y, xmin, xmax)

### Function: axhspan(ymin, ymax, xmin, xmax)

### Function: axis()

### Function: axline(xy1, xy2)

### Function: axvline(x, ymin, ymax)

### Function: axvspan(xmin, xmax, ymin, ymax)

### Function: bar(x, height, width, bottom)

### Function: barbs()

### Function: barh(y, width, height, left)

### Function: bar_label(container, labels)

### Function: boxplot(x, notch, sym, vert, orientation, whis, positions, widths, patch_artist, bootstrap, usermedians, conf_intervals, meanline, showmeans, showcaps, showbox, showfliers, boxprops, tick_labels, flierprops, medianprops, meanprops, capprops, whiskerprops, manage_ticks, autorange, zorder, capwidths, label)

### Function: broken_barh(xranges, yrange)

### Function: clabel(CS, levels)

### Function: cohere(x, y, NFFT, Fs, Fc, detrend, window, noverlap, pad_to, sides, scale_by_freq)

### Function: contour()

### Function: contourf()

### Function: csd(x, y, NFFT, Fs, Fc, detrend, window, noverlap, pad_to, sides, scale_by_freq, return_line)

### Function: ecdf(x, weights)

### Function: errorbar(x, y, yerr, xerr, fmt, ecolor, elinewidth, capsize, barsabove, lolims, uplims, xlolims, xuplims, errorevery, capthick)

### Function: eventplot(positions, orientation, lineoffsets, linelengths, linewidths, colors, alpha, linestyles)

### Function: fill()

### Function: fill_between(x, y1, y2, where, interpolate, step)

### Function: fill_betweenx(y, x1, x2, where, step, interpolate)

### Function: grid(visible, which, axis)

### Function: hexbin(x, y, C, gridsize, bins, xscale, yscale, extent, cmap, norm, vmin, vmax, alpha, linewidths, edgecolors, reduce_C_function, mincnt, marginals, colorizer)

### Function: hist(x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked)

### Function: stairs(values, edges)

### Function: hist2d(x, y, bins, range, density, weights, cmin, cmax)

### Function: hlines(y, xmin, xmax, colors, linestyles, label)

### Function: imshow(X, cmap, norm)

### Function: legend()

### Function: locator_params(axis, tight)

### Function: loglog()

### Function: magnitude_spectrum(x, Fs, Fc, window, pad_to, sides, scale)

### Function: margins()

### Function: minorticks_off()

### Function: minorticks_on()

### Function: pcolor()

### Function: pcolormesh()

### Function: phase_spectrum(x, Fs, Fc, window, pad_to, sides)

### Function: pie(x, explode, labels, colors, autopct, pctdistance, shadow, labeldistance, startangle, radius, counterclock, wedgeprops, textprops, center, frame, rotatelabels)

### Function: plot()

### Function: plot_date(x, y, fmt, tz, xdate, ydate)

### Function: psd(x, NFFT, Fs, Fc, detrend, window, noverlap, pad_to, sides, scale_by_freq, return_line)

### Function: quiver()

### Function: quiverkey(Q, X, Y, U, label)

### Function: scatter(x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths)

### Function: semilogx()

### Function: semilogy()

### Function: specgram(x, NFFT, Fs, Fc, detrend, window, noverlap, cmap, xextent, pad_to, sides, scale_by_freq, mode, scale, vmin, vmax)

### Function: spy(Z, precision, marker, markersize, aspect, origin)

### Function: stackplot(x)

### Function: stem()

### Function: step(x, y)

### Function: streamplot(x, y, u, v, density, linewidth, color, cmap, norm, arrowsize, arrowstyle, minlength, transform, zorder, start_points, maxlength, integration_direction, broken_streamlines)

### Function: table(cellText, cellColours, cellLoc, colWidths, rowLabels, rowColours, rowLoc, colLabels, colColours, colLoc, loc, bbox, edges)

### Function: text(x, y, s, fontdict)

### Function: tick_params(axis)

### Function: ticklabel_format()

### Function: tricontour()

### Function: tricontourf()

### Function: tripcolor()

### Function: triplot()

### Function: violinplot(dataset, positions, vert, orientation, widths, showmeans, showextrema, showmedians, quantiles, points, bw_method, side)

### Function: vlines(x, ymin, ymax, colors, linestyles, label)

### Function: xcorr(x, y, normed, detrend, usevlines, maxlags)

### Function: sci(im)

### Function: title(label, fontdict, loc, pad)

### Function: xlabel(xlabel, fontdict, labelpad)

### Function: ylabel(ylabel, fontdict, labelpad)

### Function: xscale(value)

### Function: yscale(value)

### Function: autumn()

**Description:** Set the colormap to 'autumn'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: bone()

**Description:** Set the colormap to 'bone'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: cool()

**Description:** Set the colormap to 'cool'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: copper()

**Description:** Set the colormap to 'copper'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: flag()

**Description:** Set the colormap to 'flag'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: gray()

**Description:** Set the colormap to 'gray'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: hot()

**Description:** Set the colormap to 'hot'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: hsv()

**Description:** Set the colormap to 'hsv'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: jet()

**Description:** Set the colormap to 'jet'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: pink()

**Description:** Set the colormap to 'pink'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: prism()

**Description:** Set the colormap to 'prism'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: spring()

**Description:** Set the colormap to 'spring'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: summer()

**Description:** Set the colormap to 'summer'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: winter()

**Description:** Set the colormap to 'winter'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: magma()

**Description:** Set the colormap to 'magma'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: inferno()

**Description:** Set the colormap to 'inferno'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: plasma()

**Description:** Set the colormap to 'plasma'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: viridis()

**Description:** Set the colormap to 'viridis'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

### Function: nipy_spectral()

**Description:** Set the colormap to 'nipy_spectral'.

This changes the default colormap as well as the colormap of the current
image if there is one. See ``help(colormaps)`` for more information.

## Class: backend_mod

### Function: new_figure_manager_given_figure(num, figure)

### Function: new_figure_manager(num)

### Function: draw_if_interactive()
