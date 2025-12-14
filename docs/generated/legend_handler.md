## AI Summary

A file named legend_handler.py.


### Function: update_from_first_child(tgt, src)

## Class: HandlerBase

**Description:** A base class for default legend handlers.

The derived classes are meant to override *create_artists* method, which
has the following signature::

  def create_artists(self, legend, orig_handle,
                     xdescent, ydescent, width, height, fontsize,
                     trans):

The overridden method needs to create artists of the given
transform that fits in the given dimension (xdescent, ydescent,
width, height) that are scaled by fontsize if necessary.

## Class: HandlerNpoints

**Description:** A legend handler that shows *numpoints* points in the legend entry.

## Class: HandlerNpointsYoffsets

**Description:** A legend handler that shows *numpoints* in the legend, and allows them to
be individually offset in the y-direction.

## Class: HandlerLine2DCompound

**Description:** Original handler for `.Line2D` instances, that relies on combining
a line-only with a marker-only artist.  May be deprecated in the future.

## Class: HandlerLine2D

**Description:** Handler for `.Line2D` instances.

See Also
--------
HandlerLine2DCompound : An earlier handler implementation, which used one
                        artist for the line and another for the marker(s).

## Class: HandlerPatch

**Description:** Handler for `.Patch` instances.

## Class: HandlerStepPatch

**Description:** Handler for `~.matplotlib.patches.StepPatch` instances.

## Class: HandlerLineCollection

**Description:** Handler for `.LineCollection` instances.

## Class: HandlerRegularPolyCollection

**Description:** Handler for `.RegularPolyCollection`\s.

## Class: HandlerPathCollection

**Description:** Handler for `.PathCollection`\s, which are used by `~.Axes.scatter`.

## Class: HandlerCircleCollection

**Description:** Handler for `.CircleCollection`\s.

## Class: HandlerErrorbar

**Description:** Handler for Errorbars.

## Class: HandlerStem

**Description:** Handler for plots produced by `~.Axes.stem`.

## Class: HandlerTuple

**Description:** Handler for Tuple.

## Class: HandlerPolyCollection

**Description:** Handler for `.PolyCollection` used in `~.Axes.fill_between` and
`~.Axes.stackplot`.

### Function: __init__(self, xpad, ypad, update_func)

**Description:** Parameters
----------
xpad : float, optional
    Padding in x-direction.
ypad : float, optional
    Padding in y-direction.
update_func : callable, optional
    Function for updating the legend handler properties from another
    legend handler, used by `~HandlerBase.update_prop`.

### Function: _update_prop(self, legend_handle, orig_handle)

### Function: _default_update_prop(self, legend_handle, orig_handle)

### Function: update_prop(self, legend_handle, orig_handle, legend)

### Function: adjust_drawing_area(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize)

### Function: legend_artist(self, legend, orig_handle, fontsize, handlebox)

**Description:** Return the artist that this HandlerBase generates for the given
original artist/handle.

Parameters
----------
legend : `~matplotlib.legend.Legend`
    The legend for which these legend artists are being created.
orig_handle : :class:`matplotlib.artist.Artist` or similar
    The object for which these legend artists are being created.
fontsize : int
    The fontsize in pixels. The artists being created should
    be scaled according to the given fontsize.
handlebox : `~matplotlib.offsetbox.OffsetBox`
    The box which has been created to hold this legend entry's
    artists. Artists created in the `legend_artist` method must
    be added to this handlebox inside this method.

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

**Description:** Return the legend artists generated.

Parameters
----------
legend : `~matplotlib.legend.Legend`
    The legend for which these legend artists are being created.
orig_handle : `~matplotlib.artist.Artist` or similar
    The object for which these legend artists are being created.
xdescent, ydescent, width, height : int
    The rectangle (*xdescent*, *ydescent*, *width*, *height*) that the
    legend artists being created should fit within.
fontsize : int
    The fontsize in pixels. The legend artists being created should
    be scaled according to the given fontsize.
trans : `~matplotlib.transforms.Transform`
    The transform that is applied to the legend artists being created.
    Typically from unit coordinates in the handler box to screen
    coordinates.

### Function: __init__(self, marker_pad, numpoints)

**Description:** Parameters
----------
marker_pad : float
    Padding between points in legend entry.
numpoints : int
    Number of points to show in legend entry.
**kwargs
    Keyword arguments forwarded to `.HandlerBase`.

### Function: get_numpoints(self, legend)

### Function: get_xdata(self, legend, xdescent, ydescent, width, height, fontsize)

### Function: __init__(self, numpoints, yoffsets)

**Description:** Parameters
----------
numpoints : int
    Number of points to show in legend entry.
yoffsets : array of floats
    Length *numpoints* list of y offsets for each point in
    legend entry.
**kwargs
    Keyword arguments forwarded to `.HandlerNpoints`.

### Function: get_ydata(self, legend, xdescent, ydescent, width, height, fontsize)

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: __init__(self, patch_func)

**Description:** Parameters
----------
patch_func : callable, optional
    The function that creates the legend key artist.
    *patch_func* should have the signature::

        def patch_func(legend=legend, orig_handle=orig_handle,
                       xdescent=xdescent, ydescent=ydescent,
                       width=width, height=height, fontsize=fontsize)

    Subsequently, the created artist will have its ``update_prop``
    method called and the appropriate transform will be applied.

**kwargs
    Keyword arguments forwarded to `.HandlerBase`.

### Function: _create_patch(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize)

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: _create_patch(orig_handle, xdescent, ydescent, width, height)

### Function: _create_line(orig_handle, width, height)

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: get_numpoints(self, legend)

### Function: _default_update_prop(self, legend_handle, orig_handle)

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: __init__(self, yoffsets, sizes)

### Function: get_numpoints(self, legend)

### Function: get_sizes(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize)

### Function: update_prop(self, legend_handle, orig_handle, legend)

### Function: create_collection(self, orig_handle, sizes, offsets, offset_transform)

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: create_collection(self, orig_handle, sizes, offsets, offset_transform)

### Function: create_collection(self, orig_handle, sizes, offsets, offset_transform)

### Function: __init__(self, xerr_size, yerr_size, marker_pad, numpoints)

### Function: get_err_size(self, legend, xdescent, ydescent, width, height, fontsize)

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: __init__(self, marker_pad, numpoints, bottom, yoffsets)

**Description:** Parameters
----------
marker_pad : float, default: 0.3
    Padding between points in legend entry.
numpoints : int, optional
    Number of points to show in legend entry.
bottom : float, optional

yoffsets : array of floats, optional
    Length *numpoints* list of y offsets for each point in
    legend entry.
**kwargs
    Keyword arguments forwarded to `.HandlerNpointsYoffsets`.

### Function: get_ydata(self, legend, xdescent, ydescent, width, height, fontsize)

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: _copy_collection_props(self, legend_handle, orig_handle)

**Description:** Copy properties from the `.LineCollection` *orig_handle* to the
`.Line2D` *legend_handle*.

### Function: __init__(self, ndivide, pad)

**Description:** Parameters
----------
ndivide : int or None, default: 1
    The number of sections to divide the legend area into.  If None,
    use the length of the input tuple.
pad : float, default: :rc:`legend.borderpad`
    Padding in units of fraction of font size.
**kwargs
    Keyword arguments forwarded to `.HandlerBase`.

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: _update_prop(self, legend_handle, orig_handle)

### Function: create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

### Function: first_color(colors)

### Function: get_first(prop_array)
