## AI Summary

A file named colorbar.py.


### Function: _set_ticks_on_axis_warn()

## Class: _ColorbarSpine

## Class: _ColorbarAxesLocator

**Description:** Shrink the Axes if there are triangular or rectangular extends.

## Class: Colorbar

**Description:** Draw a colorbar in an existing Axes.

Typically, colorbars are created using `.Figure.colorbar` or
`.pyplot.colorbar` and associated with `.ScalarMappable`\s (such as an
`.AxesImage` generated via `~.axes.Axes.imshow`).

In order to draw a colorbar not associated with other elements in the
figure, e.g. when showing a colormap by itself, one can create an empty
`.ScalarMappable`, or directly pass *cmap* and *norm* instead of *mappable*
to `Colorbar`.

Useful public methods are :meth:`set_label` and :meth:`add_lines`.

Attributes
----------
ax : `~matplotlib.axes.Axes`
    The `~.axes.Axes` instance in which the colorbar is drawn.
lines : list
    A list of `.LineCollection` (empty if no lines were drawn).
dividers : `.LineCollection`
    A LineCollection (empty if *drawedges* is ``False``).

### Function: _normalize_location_orientation(location, orientation)

### Function: _get_orientation_from_location(location)

### Function: _get_ticklocation_from_orientation(orientation)

### Function: make_axes(parents, location, orientation, fraction, shrink, aspect)

**Description:** Create an `~.axes.Axes` suitable for a colorbar.

The Axes is placed in the figure of the *parents* Axes, by resizing and
repositioning *parents*.

Parameters
----------
parents : `~matplotlib.axes.Axes` or iterable or `numpy.ndarray` of `~.axes.Axes`
    The Axes to use as parents for placing the colorbar.
%(_make_axes_kw_doc)s

Returns
-------
cax : `~matplotlib.axes.Axes`
    The child Axes.
kwargs : dict
    The reduced keyword dictionary to be passed when creating the colorbar
    instance.

### Function: make_axes_gridspec(parent)

**Description:** Create an `~.axes.Axes` suitable for a colorbar.

The Axes is placed in the figure of the *parent* Axes, by resizing and
repositioning *parent*.

This function is similar to `.make_axes` and mostly compatible with it.
Primary differences are

- `.make_axes_gridspec` requires the *parent* to have a subplotspec.
- `.make_axes` positions the Axes in figure coordinates;
  `.make_axes_gridspec` positions it using a subplotspec.
- `.make_axes` updates the position of the parent.  `.make_axes_gridspec`
  replaces the parent gridspec with a new one.

Parameters
----------
parent : `~matplotlib.axes.Axes`
    The Axes to use as parent for placing the colorbar.
%(_make_axes_kw_doc)s

Returns
-------
cax : `~matplotlib.axes.Axes`
    The child Axes.
kwargs : dict
    The reduced keyword dictionary to be passed when creating the colorbar
    instance.

### Function: __init__(self, axes)

### Function: get_window_extent(self, renderer)

### Function: set_xy(self, xy)

### Function: draw(self, renderer)

### Function: __init__(self, cbar)

### Function: __call__(self, ax, renderer)

### Function: get_subplotspec(self)

### Function: __init__(self, ax, mappable)

**Description:** Parameters
----------
ax : `~matplotlib.axes.Axes`
    The `~.axes.Axes` instance in which the colorbar is drawn.

mappable : `.ScalarMappable`
    The mappable whose colormap and norm will be used.

    To show the colors versus index instead of on a 0-1 scale, set the
    mappable's norm to ``colors.NoNorm()``.

alpha : float
    The colorbar transparency between 0 (transparent) and 1 (opaque).

location : None or {'left', 'right', 'top', 'bottom'}
    Set the colorbar's *orientation* and *ticklocation*. Colorbars on
    the left and right are vertical, colorbars at the top and bottom
    are horizontal. The *ticklocation* is the same as *location*, so if
    *location* is 'top', the ticks are on the top. *orientation* and/or
    *ticklocation* can be provided as well and overrides the value set by
    *location*, but there will be an error for incompatible combinations.

    .. versionadded:: 3.7

%(_colormap_kw_doc)s

Other Parameters
----------------
cmap : `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
    The colormap to use.  This parameter is ignored, unless *mappable* is
    None.

norm : `~matplotlib.colors.Normalize`
    The normalization to use.  This parameter is ignored, unless *mappable*
    is None.

orientation : None or {'vertical', 'horizontal'}
    If None, use the value determined by *location*. If both
    *orientation* and *location* are None then defaults to 'vertical'.

ticklocation : {'auto', 'left', 'right', 'top', 'bottom'}
    The location of the colorbar ticks. The *ticklocation* must match
    *orientation*. For example, a horizontal colorbar can only have ticks
    at the top or the bottom. If 'auto', the ticks will be the same as
    *location*, so a colorbar to the left will have ticks to the left. If
    *location* is None, the ticks will be at the bottom for a horizontal
    colorbar and at the right for a vertical.

### Function: long_axis(self)

**Description:** Axis that has decorations (ticks, etc) on it.

### Function: locator(self)

**Description:** Major tick `.Locator` for the colorbar.

### Function: locator(self, loc)

### Function: minorlocator(self)

**Description:** Minor tick `.Locator` for the colorbar.

### Function: minorlocator(self, loc)

### Function: formatter(self)

**Description:** Major tick label `.Formatter` for the colorbar.

### Function: formatter(self, fmt)

### Function: minorformatter(self)

**Description:** Minor tick `.Formatter` for the colorbar.

### Function: minorformatter(self, fmt)

### Function: _cbar_cla(self)

**Description:** Function to clear the interactive colorbar state.

### Function: update_normal(self, mappable)

**Description:** Update solid patches, lines, etc.

This is meant to be called when the norm of the image or contour plot
to which this colorbar belongs changes.

If the norm on the mappable is different than before, this resets the
locator and formatter for the axis, so if these have been customized,
they will need to be customized again.  However, if the norm only
changes values of *vmin*, *vmax* or *cmap* then the old formatter
and locator will be preserved.

### Function: _draw_all(self)

**Description:** Calculate any free parameters based on the current cmap and norm,
and do all the drawing.

### Function: _add_solids(self, X, Y, C)

**Description:** Draw the colors; optionally add separators.

### Function: _update_dividers(self)

### Function: _add_solids_patches(self, X, Y, C, mappable)

### Function: _do_extends(self, ax)

**Description:** Add the extend tri/rectangles on the outside of the Axes.

ax is unused, but required due to the callbacks on xlim/ylim changed

### Function: add_lines(self)

**Description:** Draw lines on the colorbar.

The lines are appended to the list :attr:`lines`.

Parameters
----------
levels : array-like
    The positions of the lines.
colors : :mpltype:`color` or list of :mpltype:`color`
    Either a single color applying to all lines or one color value for
    each line.
linewidths : float or array-like
    Either a single linewidth applying to all lines or one linewidth
    for each line.
erase : bool, default: True
    Whether to remove any previously added lines.

Notes
-----
Alternatively, this method can also be called with the signature
``colorbar.add_lines(contour_set, erase=True)``, in which case
*levels*, *colors*, and *linewidths* are taken from *contour_set*.

### Function: update_ticks(self)

**Description:** Set up the ticks and ticklabels. This should not be needed by users.

### Function: _get_ticker_locator_formatter(self)

**Description:** Return the ``locator`` and ``formatter`` of the colorbar.

If they have not been defined (i.e. are *None*), the formatter and
locator are retrieved from the axis, or from the value of the
boundaries for a boundary norm.

Called by update_ticks...

### Function: set_ticks(self, ticks)

**Description:** Set tick locations.

Parameters
----------
ticks : 1D array-like
    List of tick locations.
labels : list of str, optional
    List of tick labels. If not set, the labels show the data value.
minor : bool, default: False
    If ``False``, set the major ticks; if ``True``, the minor ticks.
**kwargs
    `.Text` properties for the labels. These take effect only if you
    pass *labels*. In other cases, please use `~.Axes.tick_params`.

### Function: get_ticks(self, minor)

**Description:** Return the ticks as a list of locations.

Parameters
----------
minor : boolean, default: False
    if True return the minor ticks.

### Function: set_ticklabels(self, ticklabels)

**Description:** [*Discouraged*] Set tick labels.

.. admonition:: Discouraged

    The use of this method is discouraged, because of the dependency
    on tick positions. In most cases, you'll want to use
    ``set_ticks(positions, labels=labels)`` instead.

    If you are using this method, you should always fix the tick
    positions before, e.g. by using `.Colorbar.set_ticks` or by
    explicitly setting a `~.ticker.FixedLocator` on the long axis
    of the colorbar. Otherwise, ticks are free to move and the
    labels may end up in unexpected positions.

Parameters
----------
ticklabels : sequence of str or of `.Text`
    Texts for labeling each tick location in the sequence set by
    `.Colorbar.set_ticks`; the number of labels must match the number
    of locations.

update_ticks : bool, default: True
    This keyword argument is ignored and will be removed.
    Deprecated

minor : bool
    If True, set minor ticks instead of major ticks.

**kwargs
    `.Text` properties for the labels.

### Function: minorticks_on(self)

**Description:** Turn on colorbar minor ticks.

### Function: minorticks_off(self)

**Description:** Turn the minor ticks of the colorbar off.

### Function: set_label(self, label)

**Description:** Add a label to the long axis of the colorbar.

Parameters
----------
label : str
    The label text.
loc : str, optional
    The location of the label.

    - For horizontal orientation one of {'left', 'center', 'right'}
    - For vertical orientation one of {'bottom', 'center', 'top'}

    Defaults to :rc:`xaxis.labellocation` or :rc:`yaxis.labellocation`
    depending on the orientation.
**kwargs
    Keyword arguments are passed to `~.Axes.set_xlabel` /
    `~.Axes.set_ylabel`.
    Supported keywords are *labelpad* and `.Text` properties.

### Function: set_alpha(self, alpha)

**Description:** Set the transparency between 0 (transparent) and 1 (opaque).

If an array is provided, *alpha* will be set to None to use the
transparency values associated with the colormap.

### Function: _set_scale(self, scale)

**Description:** Set the colorbar long axis scale.

Parameters
----------
scale : {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
    The axis scale type to apply.

**kwargs
    Different keyword arguments are accepted, depending on the scale.
    See the respective class keyword arguments:

    - `matplotlib.scale.LinearScale`
    - `matplotlib.scale.LogScale`
    - `matplotlib.scale.SymmetricalLogScale`
    - `matplotlib.scale.LogitScale`
    - `matplotlib.scale.FuncScale`
    - `matplotlib.scale.AsinhScale`

Notes
-----
By default, Matplotlib supports the above-mentioned scales.
Additionally, custom scales may be registered using
`matplotlib.scale.register_scale`. These scales can then also
be used here.

### Function: remove(self)

**Description:** Remove this colorbar from the figure.

If the colorbar was created with ``use_gridspec=True`` the previous
gridspec is restored.

### Function: _process_values(self)

**Description:** Set `_boundaries` and `_values` based on the self.boundaries and
self.values if not None, or based on the size of the colormap and
the vmin/vmax of the norm.

### Function: _mesh(self)

**Description:** Return the coordinate arrays for the colorbar pcolormesh/patches.

These are scaled between vmin and vmax, and already handle colorbar
orientation.

### Function: _forward_boundaries(self, x)

### Function: _inverse_boundaries(self, x)

### Function: _reset_locator_formatter_scale(self)

**Description:** Reset the locator et al to defaults.  Any user-hardcoded changes
need to be re-entered if this gets called (either at init, or when
the mappable normal gets changed: Colorbar.update_normal)

### Function: _locate(self, x)

**Description:** Given a set of color data values, return their
corresponding colorbar data coordinates.

### Function: _uniform_y(self, N)

**Description:** Return colorbar data coordinates for *N* uniformly
spaced boundaries, plus extension lengths if required.

### Function: _proportional_y(self)

**Description:** Return colorbar data coordinates for the boundaries of
a proportional colorbar, plus extension lengths if required:

### Function: _get_extension_lengths(self, frac, automin, automax, default)

**Description:** Return the lengths of colorbar extensions.

This is a helper method for _uniform_y and _proportional_y.

### Function: _extend_lower(self)

**Description:** Return whether the lower limit is open ended.

### Function: _extend_upper(self)

**Description:** Return whether the upper limit is open ended.

### Function: _short_axis(self)

**Description:** Return the short axis

### Function: _get_view(self)

### Function: _set_view(self, view)

### Function: _set_view_from_bbox(self, bbox, direction, mode, twinx, twiny)

### Function: drag_pan(self, button, key, x, y)
