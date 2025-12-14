## AI Summary

A file named offsetbox.py.


### Function: _compat_get_offset(meth)

**Description:** Decorator for the get_offset method of OffsetBox and subclasses, that
allows supporting both the new signature (self, bbox, renderer) and the old
signature (self, width, height, xdescent, ydescent, renderer).

### Function: _bbox_artist()

### Function: _get_packed_offsets(widths, total, sep, mode)

**Description:** Pack boxes specified by their *widths*.

For simplicity of the description, the terminology used here assumes a
horizontal layout, but the function works equally for a vertical layout.

There are three packing *mode*\s:

- 'fixed': The elements are packed tight to the left with a spacing of
  *sep* in between. If *total* is *None* the returned total will be the
  right edge of the last box. A non-*None* total will be passed unchecked
  to the output. In particular this means that right edge of the last
  box may be further to the right than the returned total.

- 'expand': Distribute the boxes with equal spacing so that the left edge
  of the first box is at 0, and the right edge of the last box is at
  *total*. The parameter *sep* is ignored in this mode. A total of *None*
  is accepted and considered equal to 1. The total is returned unchanged
  (except for the conversion *None* to 1). If the total is smaller than
  the sum of the widths, the laid out boxes will overlap.

- 'equal': If *total* is given, the total space is divided in N equal
  ranges and each box is left-aligned within its subspace.
  Otherwise (*total* is *None*), *sep* must be provided and each box is
  left-aligned in its subspace of width ``(max(widths) + sep)``. The
  total width is then calculated to be ``N * (max(widths) + sep)``.

Parameters
----------
widths : list of float
    Widths of boxes to be packed.
total : float or None
    Intended total length. *None* if not used.
sep : float or None
    Spacing between boxes.
mode : {'fixed', 'expand', 'equal'}
    The packing mode.

Returns
-------
total : float
    The total width needed to accommodate the laid out boxes.
offsets : array of float
    The left offsets of the boxes.

### Function: _get_aligned_offsets(yspans, height, align)

**Description:** Align boxes each specified by their ``(y0, y1)`` spans.

For simplicity of the description, the terminology used here assumes a
horizontal layout (i.e., vertical alignment), but the function works
equally for a vertical layout.

Parameters
----------
yspans
    List of (y0, y1) spans of boxes to be aligned.
height : float or None
    Intended total height. If None, the maximum of the heights
    (``y1 - y0``) in *yspans* is used.
align : {'baseline', 'left', 'top', 'right', 'bottom', 'center'}
    The alignment anchor of the boxes.

Returns
-------
(y0, y1)
    y range spanned by the packing.  If a *height* was originally passed
    in, then for all alignments other than "baseline", a span of ``(0,
    height)`` is used without checking that it is actually large enough).
descent
    The descent of the packing.
offsets
    The bottom offsets of the boxes.

## Class: OffsetBox

**Description:** A simple container artist.

The child artists are meant to be drawn at a relative position to its
parent.

Being an artist itself, all parameters are passed on to `.Artist`.

## Class: PackerBase

## Class: VPacker

**Description:** VPacker packs its children vertically, automatically adjusting their
relative positions at draw time.

.. code-block:: none

   +---------+
   | Child 1 |
   | Child 2 |
   | Child 3 |
   +---------+

## Class: HPacker

**Description:** HPacker packs its children horizontally, automatically adjusting their
relative positions at draw time.

.. code-block:: none

   +-------------------------------+
   | Child 1    Child 2    Child 3 |
   +-------------------------------+

## Class: PaddedBox

**Description:** A container to add a padding around an `.Artist`.

The `.PaddedBox` contains a `.FancyBboxPatch` that is used to visualize
it when rendering.

.. code-block:: none

   +----------------------------+
   |                            |
   |                            |
   |                            |
   | <--pad--> Artist           |
   |             ^              |
   |            pad             |
   |             v              |
   +----------------------------+

Attributes
----------
pad : float
    The padding in points.
patch : `.FancyBboxPatch`
    When *draw_frame* is True, this `.FancyBboxPatch` is made visible and
    creates a border around the box.

## Class: DrawingArea

**Description:** The DrawingArea can contain any Artist as a child. The DrawingArea
has a fixed width and height. The position of children relative to
the parent is fixed. The children can be clipped at the
boundaries of the parent.

## Class: TextArea

**Description:** The TextArea is a container artist for a single Text instance.

The text is placed at (0, 0) with baseline+left alignment, by default. The
width and height of the TextArea instance is the width and height of its
child text.

## Class: AuxTransformBox

**Description:** An OffsetBox with an auxiliary transform.

All child artists are first transformed with *aux_transform*, then
translated with an offset (the same for all children) so the bounding
box of the children matches the drawn box.  (In other words, adding an
arbitrary translation to *aux_transform* has no effect as it will be
cancelled out by the later offsetting.)

`AuxTransformBox` is similar to `.DrawingArea`, except that the extent of
the box is not predetermined but calculated from the window extent of its
children, and the extent of the children will be calculated in the
transformed coordinate.

## Class: AnchoredOffsetbox

**Description:** An OffsetBox placed according to location *loc*.

AnchoredOffsetbox has a single child.  When multiple children are needed,
use an extra OffsetBox to enclose them.  By default, the offset box is
anchored against its parent Axes. You may explicitly specify the
*bbox_to_anchor*.

### Function: _get_anchored_bbox(loc, bbox, parentbbox, borderpad)

**Description:** Return the (x, y) position of the *bbox* anchored at the *parentbbox* with
the *loc* code with the *borderpad*.

## Class: AnchoredText

**Description:** AnchoredOffsetbox with Text.

## Class: OffsetImage

## Class: AnnotationBbox

**Description:** Container for an `OffsetBox` referring to a specific position *xy*.

Optionally an arrow pointing from the offsetbox to *xy* can be drawn.

This is like `.Annotation`, but with `OffsetBox` instead of `.Text`.

## Class: DraggableBase

**Description:** Helper base class for a draggable artist (legend, offsetbox).

Derived classes must override the following methods::

    def save_offset(self):
        '''
        Called when the object is picked for dragging; should save the
        reference position of the artist.
        '''

    def update_offset(self, dx, dy):
        '''
        Called during the dragging; (*dx*, *dy*) is the pixel offset from
        the point where the mouse drag started.
        '''

Optionally, you may override the following method::

    def finalize_offset(self):
        '''Called when the mouse is released.'''

In the current implementation of `.DraggableLegend` and
`DraggableAnnotation`, `update_offset` places the artists in display
coordinates, and `finalize_offset` recalculates their position in axes
coordinate and set a relevant attribute.

## Class: DraggableOffsetBox

## Class: DraggableAnnotation

### Function: get_offset(self)

### Function: __init__(self)

### Function: set_figure(self, fig)

**Description:** Set the `.Figure` for the `.OffsetBox` and all its children.

Parameters
----------
fig : `~matplotlib.figure.Figure`

### Function: axes(self, ax)

### Function: contains(self, mouseevent)

**Description:** Delegate the mouse event contains-check to the children.

As a container, the `.OffsetBox` does not respond itself to
mouseevents.

Parameters
----------
mouseevent : `~matplotlib.backend_bases.MouseEvent`

Returns
-------
contains : bool
    Whether any values are within the radius.
details : dict
    An artist-specific dictionary of details of the event context,
    such as which points are contained in the pick radius. See the
    individual Artist subclasses for details.

See Also
--------
.Artist.contains

### Function: set_offset(self, xy)

**Description:** Set the offset.

Parameters
----------
xy : (float, float) or callable
    The (x, y) coordinates of the offset in display units. These can
    either be given explicitly as a tuple (x, y), or by providing a
    function that converts the extent into the offset. This function
    must have the signature::

        def offset(width, height, xdescent, ydescent, renderer) -> (float, float)

### Function: get_offset(self, bbox, renderer)

**Description:** Return the offset as a tuple (x, y).

The extent parameters have to be provided to handle the case where the
offset is dynamically determined by a callable (see
`~.OffsetBox.set_offset`).

Parameters
----------
bbox : `.Bbox`
renderer : `.RendererBase` subclass

### Function: set_width(self, width)

**Description:** Set the width of the box.

Parameters
----------
width : float

### Function: set_height(self, height)

**Description:** Set the height of the box.

Parameters
----------
height : float

### Function: get_visible_children(self)

**Description:** Return a list of the visible child `.Artist`\s.

### Function: get_children(self)

**Description:** Return a list of the child `.Artist`\s.

### Function: _get_bbox_and_child_offsets(self, renderer)

**Description:** Return the bbox of the offsetbox and the child offsets.

The bbox should satisfy ``x0 <= x1 and y0 <= y1``.

Parameters
----------
renderer : `.RendererBase` subclass

Returns
-------
bbox
list of (xoffset, yoffset) pairs

### Function: get_bbox(self, renderer)

**Description:** Return the bbox of the offsetbox, ignoring parent offsets.

### Function: get_window_extent(self, renderer)

### Function: draw(self, renderer)

**Description:** Update the location of children if necessary and draw them
to the given *renderer*.

### Function: __init__(self, pad, sep, width, height, align, mode, children)

**Description:** Parameters
----------
pad : float, default: 0.0
    The boundary padding in points.

sep : float, default: 0.0
    The spacing between items in points.

width, height : float, optional
    Width and height of the container box in pixels, calculated if
    *None*.

align : {'top', 'bottom', 'left', 'right', 'center', 'baseline'}, default: 'baseline'
    Alignment of boxes.

mode : {'fixed', 'expand', 'equal'}, default: 'fixed'
    The packing mode.

    - 'fixed' packs the given `.Artist`\s tight with *sep* spacing.
    - 'expand' uses the maximal available space to distribute the
      artists with equal spacing in between.
    - 'equal': Each artist an equal fraction of the available space
      and is left-aligned (or top-aligned) therein.

children : list of `.Artist`
    The artists to pack.

Notes
-----
*pad* and *sep* are in points and will be scaled with the renderer
dpi, while *width* and *height* are in pixels.

### Function: _get_bbox_and_child_offsets(self, renderer)

### Function: _get_bbox_and_child_offsets(self, renderer)

### Function: __init__(self, child, pad)

**Description:** Parameters
----------
child : `~matplotlib.artist.Artist`
    The contained `.Artist`.
pad : float, default: 0.0
    The padding in points. This will be scaled with the renderer dpi.
    In contrast, *width* and *height* are in *pixels* and thus not
    scaled.
draw_frame : bool
    Whether to draw the contained `.FancyBboxPatch`.
patch_attrs : dict or None
    Additional parameters passed to the contained `.FancyBboxPatch`.

### Function: _get_bbox_and_child_offsets(self, renderer)

### Function: draw(self, renderer)

### Function: update_frame(self, bbox, fontsize)

### Function: draw_frame(self, renderer)

### Function: __init__(self, width, height, xdescent, ydescent, clip)

**Description:** Parameters
----------
width, height : float
    Width and height of the container box.
xdescent, ydescent : float
    Descent of the box in x- and y-direction.
clip : bool
    Whether to clip the children to the box.

### Function: clip_children(self)

**Description:** If the children of this DrawingArea should be clipped
by DrawingArea bounding box.

### Function: clip_children(self, val)

### Function: get_transform(self)

**Description:** Return the `~matplotlib.transforms.Transform` applied to the children.

### Function: set_transform(self, t)

**Description:** set_transform is ignored.

### Function: set_offset(self, xy)

**Description:** Set the offset of the container.

Parameters
----------
xy : (float, float)
    The (x, y) coordinates of the offset in display units.

### Function: get_offset(self)

**Description:** Return offset of the container.

### Function: get_bbox(self, renderer)

### Function: add_artist(self, a)

**Description:** Add an `.Artist` to the container box.

### Function: draw(self, renderer)

### Function: __init__(self, s)

**Description:** Parameters
----------
s : str
    The text to be displayed.
textprops : dict, default: {}
    Dictionary of keyword parameters to be passed to the `.Text`
    instance in the TextArea.
multilinebaseline : bool, default: False
    Whether the baseline for multiline text is adjusted so that it
    is (approximately) center-aligned with single-line text.

### Function: set_text(self, s)

**Description:** Set the text of this area as a string.

### Function: get_text(self)

**Description:** Return the string representation of this area's text.

### Function: set_multilinebaseline(self, t)

**Description:** Set multilinebaseline.

If True, the baseline for multiline text is adjusted so that it is
(approximately) center-aligned with single-line text.  This is used
e.g. by the legend implementation so that single-line labels are
baseline-aligned, but multiline labels are "center"-aligned with them.

### Function: get_multilinebaseline(self)

**Description:** Get multilinebaseline.

### Function: set_transform(self, t)

**Description:** set_transform is ignored.

### Function: set_offset(self, xy)

**Description:** Set the offset of the container.

Parameters
----------
xy : (float, float)
    The (x, y) coordinates of the offset in display units.

### Function: get_offset(self)

**Description:** Return offset of the container.

### Function: get_bbox(self, renderer)

### Function: draw(self, renderer)

### Function: __init__(self, aux_transform)

### Function: add_artist(self, a)

**Description:** Add an `.Artist` to the container box.

### Function: get_transform(self)

**Description:** Return the `.Transform` applied to the children.

### Function: set_transform(self, t)

**Description:** set_transform is ignored.

### Function: set_offset(self, xy)

**Description:** Set the offset of the container.

Parameters
----------
xy : (float, float)
    The (x, y) coordinates of the offset in display units.

### Function: get_offset(self)

**Description:** Return offset of the container.

### Function: get_bbox(self, renderer)

### Function: draw(self, renderer)

### Function: __init__(self, loc)

**Description:** Parameters
----------
loc : str
    The box location.  Valid locations are
    'upper left', 'upper center', 'upper right',
    'center left', 'center', 'center right',
    'lower left', 'lower center', 'lower right'.
    For backward compatibility, numeric values are accepted as well.
    See the parameter *loc* of `.Legend` for details.
pad : float, default: 0.4
    Padding around the child as fraction of the fontsize.
borderpad : float, default: 0.5
    Padding between the offsetbox frame and the *bbox_to_anchor*.
child : `.OffsetBox`
    The box that will be anchored.
prop : `.FontProperties`
    This is only used as a reference for paddings. If not given,
    :rc:`legend.fontsize` is used.
frameon : bool
    Whether to draw a frame around the box.
bbox_to_anchor : `.BboxBase`, 2-tuple, or 4-tuple of floats
    Box that is used to position the legend in conjunction with *loc*.
bbox_transform : None or :class:`matplotlib.transforms.Transform`
    The transform for the bounding box (*bbox_to_anchor*).
**kwargs
    All other parameters are passed on to `.OffsetBox`.

Notes
-----
See `.Legend` for a detailed description of the anchoring mechanism.

### Function: set_child(self, child)

**Description:** Set the child to be anchored.

### Function: get_child(self)

**Description:** Return the child.

### Function: get_children(self)

**Description:** Return the list of children.

### Function: get_bbox(self, renderer)

### Function: get_bbox_to_anchor(self)

**Description:** Return the bbox that the box is anchored to.

### Function: set_bbox_to_anchor(self, bbox, transform)

**Description:** Set the bbox that the box is anchored to.

*bbox* can be a Bbox instance, a list of [left, bottom, width,
height], or a list of [left, bottom] where the width and
height will be assumed to be zero. The bbox will be
transformed to display coordinate by the given transform.

### Function: get_offset(self, bbox, renderer)

### Function: update_frame(self, bbox, fontsize)

### Function: draw(self, renderer)

### Function: __init__(self, s, loc)

**Description:** Parameters
----------
s : str
    Text.

loc : str
    Location code. See `AnchoredOffsetbox`.

pad : float, default: 0.4
    Padding around the text as fraction of the fontsize.

borderpad : float, default: 0.5
    Spacing between the offsetbox frame and the *bbox_to_anchor*.

prop : dict, optional
    Dictionary of keyword parameters to be passed to the
    `~matplotlib.text.Text` instance contained inside AnchoredText.

**kwargs
    All other parameters are passed to `AnchoredOffsetbox`.

### Function: __init__(self, arr)

### Function: set_data(self, arr)

### Function: get_data(self)

### Function: set_zoom(self, zoom)

### Function: get_zoom(self)

### Function: get_offset(self)

**Description:** Return offset of the container.

### Function: get_children(self)

### Function: get_bbox(self, renderer)

### Function: draw(self, renderer)

### Function: __str__(self)

### Function: __init__(self, offsetbox, xy, xybox, xycoords, boxcoords)

**Description:** Parameters
----------
offsetbox : `OffsetBox`

xy : (float, float)
    The point *(x, y)* to annotate. The coordinate system is determined
    by *xycoords*.

xybox : (float, float), default: *xy*
    The position *(x, y)* to place the text at. The coordinate system
    is determined by *boxcoords*.

xycoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: 'data'
    The coordinate system that *xy* is given in. See the parameter
    *xycoords* in `.Annotation` for a detailed description.

boxcoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: value of *xycoords*
    The coordinate system that *xybox* is given in. See the parameter
    *textcoords* in `.Annotation` for a detailed description.

frameon : bool, default: True
    By default, the text is surrounded by a white `.FancyBboxPatch`
    (accessible as the ``patch`` attribute of the `.AnnotationBbox`).
    If *frameon* is set to False, this patch is made invisible.

annotation_clip: bool or None, default: None
    Whether to clip (i.e. not draw) the annotation when the annotation
    point *xy* is outside the Axes area.

    - If *True*, the annotation will be clipped when *xy* is outside
      the Axes.
    - If *False*, the annotation will always be drawn.
    - If *None*, the annotation will be clipped when *xy* is outside
      the Axes and *xycoords* is 'data'.

pad : float, default: 0.4
    Padding around the offsetbox.

box_alignment : (float, float)
    A tuple of two floats for a vertical and horizontal alignment of
    the offset box w.r.t. the *boxcoords*.
    The lower-left corner is (0, 0) and upper-right corner is (1, 1).

bboxprops : dict, optional
    A dictionary of properties to set for the annotation bounding box,
    for example *boxstyle* and *alpha*.  See `.FancyBboxPatch` for
    details.

arrowprops: dict, optional
    Arrow properties, see `.Annotation` for description.

fontsize: float or str, optional
    Translated to points and passed as *mutation_scale* into
    `.FancyBboxPatch` to scale attributes of the box style (e.g. pad
    or rounding_size).  The name is chosen in analogy to `.Text` where
    *fontsize* defines the mutation scale as well.  If not given,
    :rc:`legend.fontsize` is used.  See `.Text.set_fontsize` for valid
    values.

**kwargs
    Other `AnnotationBbox` properties.  See `.AnnotationBbox.set` for
    a list.

### Function: xyann(self)

### Function: xyann(self, xyann)

### Function: anncoords(self)

### Function: anncoords(self, coords)

### Function: contains(self, mouseevent)

### Function: get_children(self)

### Function: set_figure(self, fig)

### Function: set_fontsize(self, s)

**Description:** Set the fontsize in points.

If *s* is not given, reset to :rc:`legend.fontsize`.

### Function: get_fontsize(self)

**Description:** Return the fontsize in points.

### Function: get_window_extent(self, renderer)

### Function: get_tightbbox(self, renderer)

### Function: update_positions(self, renderer)

**Description:** Update pixel positions for the annotated point, the text, and the arrow.

### Function: draw(self, renderer)

### Function: __init__(self, ref_artist, use_blit)

### Function: _picker(artist, mouseevent)

### Function: on_motion(self, evt)

### Function: on_pick(self, evt)

### Function: on_release(self, event)

### Function: _check_still_parented(self)

### Function: disconnect(self)

**Description:** Disconnect the callbacks.

### Function: save_offset(self)

### Function: update_offset(self, dx, dy)

### Function: finalize_offset(self)

### Function: __init__(self, ref_artist, offsetbox, use_blit)

### Function: save_offset(self)

### Function: update_offset(self, dx, dy)

### Function: get_loc_in_canvas(self)

### Function: __init__(self, annotation, use_blit)

### Function: save_offset(self)

### Function: update_offset(self, dx, dy)
