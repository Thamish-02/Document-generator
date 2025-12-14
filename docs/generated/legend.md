## AI Summary

A file named legend.py.


## Class: DraggableLegend

## Class: Legend

**Description:** Place a legend on the figure/axes.

### Function: _get_legend_handles(axs, legend_handler_map)

**Description:** Yield artists that can be used as handles in a legend.

### Function: _get_legend_handles_labels(axs, legend_handler_map)

**Description:** Return handles and labels for legend.

### Function: _parse_legend_args(axs)

**Description:** Get the handles and labels from the calls to either ``figure.legend``
or ``axes.legend``.

The parser is a bit involved because we support::

    legend()
    legend(labels)
    legend(handles, labels)
    legend(labels=labels)
    legend(handles=handles)
    legend(handles=handles, labels=labels)

The behavior for a mixture of positional and keyword handles and labels
is undefined and issues a warning; it will be an error in the future.

Parameters
----------
axs : list of `.Axes`
    If handles are not given explicitly, the artists in these Axes are
    used as handles.
*args : tuple
    Positional parameters passed to ``legend()``.
handles
    The value of the keyword argument ``legend(handles=...)``, or *None*
    if that keyword argument was not used.
labels
    The value of the keyword argument ``legend(labels=...)``, or *None*
    if that keyword argument was not used.
**kwargs
    All other keyword arguments passed to ``legend()``.

Returns
-------
handles : list of (`.Artist` or tuple of `.Artist`)
    The legend handles.
labels : list of str
    The legend labels.
kwargs : dict
    *kwargs* with keywords handles and labels removed.

### Function: __init__(self, legend, use_blit, update)

**Description:** Wrapper around a `.Legend` to support mouse dragging.

Parameters
----------
legend : `.Legend`
    The `.Legend` instance to wrap.
use_blit : bool, optional
    Use blitting for faster image composition. For details see
    :ref:`func-animation`.
update : {'loc', 'bbox'}, optional
    If "loc", update the *loc* parameter of the legend upon finalizing.
    If "bbox", update the *bbox_to_anchor* parameter.

### Function: finalize_offset(self)

### Function: _update_loc(self, loc_in_canvas)

### Function: _update_bbox_to_anchor(self, loc_in_canvas)

### Function: __str__(self)

### Function: __init__(self, parent, handles, labels)

**Description:** Parameters
----------
parent : `~matplotlib.axes.Axes` or `.Figure`
    The artist that contains the legend.

handles : list of (`.Artist` or tuple of `.Artist`)
    A list of Artists (lines, patches) to be added to the legend.

labels : list of str
    A list of labels to show next to the artists. The length of handles
    and labels should be the same. If they are not, they are truncated
    to the length of the shorter list.

Other Parameters
----------------
%(_legend_kw_doc)s

Attributes
----------
legend_handles
    List of `.Artist` objects added as legend entries.

    .. versionadded:: 3.7

### Function: _set_artist_props(self, a)

**Description:** Set the boilerplate props for artists added to Axes.

### Function: set_loc(self, loc)

**Description:** Set the location of the legend.

.. versionadded:: 3.8

Parameters
----------
%(_legend_kw_set_loc_doc)s

### Function: _set_loc(self, loc)

### Function: set_ncols(self, ncols)

**Description:** Set the number of columns.

### Function: _get_loc(self)

### Function: _findoffset(self, width, height, xdescent, ydescent, renderer)

**Description:** Helper function to locate the legend.

### Function: draw(self, renderer)

### Function: get_default_handler_map(cls)

**Description:** Return the global default handler map, shared by all legends.

### Function: set_default_handler_map(cls, handler_map)

**Description:** Set the global default handler map, shared by all legends.

### Function: update_default_handler_map(cls, handler_map)

**Description:** Update the global default handler map, shared by all legends.

### Function: get_legend_handler_map(self)

**Description:** Return this legend instance's handler map.

### Function: get_legend_handler(legend_handler_map, orig_handle)

**Description:** Return a legend handler from *legend_handler_map* that
corresponds to *orig_handler*.

*legend_handler_map* should be a dictionary object (that is
returned by the get_legend_handler_map method).

It first checks if the *orig_handle* itself is a key in the
*legend_handler_map* and return the associated value.
Otherwise, it checks for each of the classes in its
method-resolution-order. If no matching key is found, it
returns ``None``.

### Function: _init_legend_box(self, handles, labels, markerfirst)

**Description:** Initialize the legend_box. The legend_box is an instance of
the OffsetBox, which is packed with legend handles and
texts. Once packed, their location is calculated during the
drawing time.

### Function: _auto_legend_data(self, renderer)

**Description:** Return display coordinates for hit testing for "best" positioning.

Returns
-------
bboxes
    List of bounding boxes of all patches.
lines
    List of `.Path` corresponding to each line.
offsets
    List of (x, y) offsets of all collection.

### Function: get_children(self)

### Function: get_frame(self)

**Description:** Return the `~.patches.Rectangle` used to frame the legend.

### Function: get_lines(self)

**Description:** Return the list of `~.lines.Line2D`\s in the legend.

### Function: get_patches(self)

**Description:** Return the list of `~.patches.Patch`\s in the legend.

### Function: get_texts(self)

**Description:** Return the list of `~.text.Text`\s in the legend.

### Function: set_alignment(self, alignment)

**Description:** Set the alignment of the legend title and the box of entries.

The entries are aligned as a single block, so that markers always
lined up.

Parameters
----------
alignment : {'center', 'left', 'right'}.

### Function: get_alignment(self)

**Description:** Get the alignment value of the legend box

### Function: set_title(self, title, prop)

**Description:** Set legend title and title style.

Parameters
----------
title : str
    The legend title.

prop : `.font_manager.FontProperties` or `str` or `pathlib.Path`
    The font properties of the legend title.
    If a `str`, it is interpreted as a fontconfig pattern parsed by
    `.FontProperties`.  If a `pathlib.Path`, it is interpreted as the
    absolute path to a font file.

### Function: get_title(self)

**Description:** Return the `.Text` instance for the legend title.

### Function: get_window_extent(self, renderer)

### Function: get_tightbbox(self, renderer)

### Function: get_frame_on(self)

**Description:** Get whether the legend box patch is drawn.

### Function: set_frame_on(self, b)

**Description:** Set whether the legend box patch is drawn.

Parameters
----------
b : bool

### Function: get_bbox_to_anchor(self)

**Description:** Return the bbox that the legend will be anchored to.

### Function: set_bbox_to_anchor(self, bbox, transform)

**Description:** Set the bbox that the legend will be anchored to.

Parameters
----------
bbox : `~matplotlib.transforms.BboxBase` or tuple
    The bounding box can be specified in the following ways:

    - A `.BboxBase` instance
    - A tuple of ``(left, bottom, width, height)`` in the given
      transform (normalized axes coordinate if None)
    - A tuple of ``(left, bottom)`` where the width and height will be
      assumed to be zero.
    - *None*, to remove the bbox anchoring, and use the parent bbox.

transform : `~matplotlib.transforms.Transform`, optional
    A transform to apply to the bounding box. If not specified, this
    will use a transform to the bounding box of the parent.

### Function: _get_anchored_bbox(self, loc, bbox, parentbbox, renderer)

**Description:** Place the *bbox* inside the *parentbbox* according to a given
location code. Return the (x, y) coordinate of the bbox.

Parameters
----------
loc : int
    A location code in range(1, 11). This corresponds to the possible
    values for ``self._loc``, excluding "best".
bbox : `~matplotlib.transforms.Bbox`
    bbox to be placed, in display coordinates.
parentbbox : `~matplotlib.transforms.Bbox`
    A parent box which will contain the bbox, in display coordinates.

### Function: _find_best_position(self, width, height, renderer)

**Description:** Determine the best location to place the legend.

### Function: contains(self, mouseevent)

### Function: set_draggable(self, state, use_blit, update)

**Description:** Enable or disable mouse dragging support of the legend.

Parameters
----------
state : bool
    Whether mouse dragging is enabled.
use_blit : bool, optional
    Use blitting for faster image composition. For details see
    :ref:`func-animation`.
update : {'loc', 'bbox'}, optional
    The legend parameter to be changed when dragged:

    - 'loc': update the *loc* parameter of the legend
    - 'bbox': update the *bbox_to_anchor* parameter of the legend

Returns
-------
`.DraggableLegend` or *None*
    If *state* is ``True`` this returns the `.DraggableLegend` helper
    instance. Otherwise this returns *None*.

### Function: get_draggable(self)

**Description:** Return ``True`` if the legend is draggable, ``False`` otherwise.
