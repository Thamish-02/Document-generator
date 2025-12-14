## AI Summary

A file named collections.py.


## Class: Collection

**Description:** Base class for Collections. Must be subclassed to be usable.

A Collection represents a sequence of `.Patch`\es that can be drawn
more efficiently together than individually. For example, when a single
path is being drawn repeatedly at different offsets, the renderer can
typically execute a ``draw_marker()`` call much more efficiently than a
series of repeated calls to ``draw_path()`` with the offsets put in
one-by-one.

Most properties of a collection can be configured per-element. Therefore,
Collections have "plural" versions of many of the properties of a `.Patch`
(e.g. `.Collection.get_paths` instead of `.Patch.get_path`). Exceptions are
the *zorder*, *hatch*, *pickradius*, *capstyle* and *joinstyle* properties,
which can only be set globally for the whole collection.

Besides these exceptions, all properties can be specified as single values
(applying to all elements) or sequences of values. The property of the
``i``\th element of the collection is::

  prop[i % len(prop)]

Each Collection can optionally be used as its own `.ScalarMappable` by
passing the *norm* and *cmap* parameters to its constructor. If the
Collection's `.ScalarMappable` matrix ``_A`` has been set (via a call
to `.Collection.set_array`), then at draw time this internal scalar
mappable will be used to set the ``facecolors`` and ``edgecolors``,
ignoring those that were manually passed in.

## Class: _CollectionWithSizes

**Description:** Base class for collections that have an array of sizes.

## Class: PathCollection

**Description:** A collection of `~.path.Path`\s, as created by e.g. `~.Axes.scatter`.

## Class: PolyCollection

## Class: FillBetweenPolyCollection

**Description:** `.PolyCollection` that fills the area between two x- or y-curves.

## Class: RegularPolyCollection

**Description:** A collection of n-sided regular polygons.

## Class: StarPolygonCollection

**Description:** Draw a collection of regular stars with *numsides* points.

## Class: AsteriskPolygonCollection

**Description:** Draw a collection of regular asterisks with *numsides* points.

## Class: LineCollection

**Description:** Represents a sequence of `.Line2D`\s that should be drawn together.

This class extends `.Collection` to represent a sequence of
`.Line2D`\s instead of just a sequence of `.Patch`\s.
Just as in `.Collection`, each property of a *LineCollection* may be either
a single value or a list of values. This list is then used cyclically for
each element of the LineCollection, so the property of the ``i``\th element
of the collection is::

  prop[i % len(prop)]

The properties of each member of a *LineCollection* default to their values
in :rc:`lines.*` instead of :rc:`patch.*`, and the property *colors* is
added in place of *edgecolors*.

## Class: EventCollection

**Description:** A collection of locations along a single axis at which an "event" occurred.

The events are given by a 1-dimensional array. They do not have an
amplitude and are displayed as parallel lines.

## Class: CircleCollection

**Description:** A collection of circles, drawn using splines.

## Class: EllipseCollection

**Description:** A collection of ellipses, drawn using splines.

## Class: PatchCollection

**Description:** A generic collection of patches.

PatchCollection draws faster than a large number of equivalent individual
Patches. It also makes it easier to assign a colormap to a heterogeneous
collection of patches.

## Class: TriMesh

**Description:** Class for the efficient drawing of a triangular mesh using Gouraud shading.

A triangular mesh is a `~matplotlib.tri.Triangulation` object.

## Class: _MeshData

**Description:** Class for managing the two dimensional coordinates of Quadrilateral meshes
and the associated data with them. This class is a mixin and is intended to
be used with another collection that will implement the draw separately.

A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
defined by the vertices ::

           (m+1, n) ----------- (m+1, n+1)
              /                   /
             /                 /
            /               /
        (m, n) -------- (m, n+1)

The mesh need not be regular and the polygons need not be convex.

Parameters
----------
coordinates : (M+1, N+1, 2) array-like
    The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
    of vertex (m, n).

shading : {'flat', 'gouraud'}, default: 'flat'

## Class: QuadMesh

**Description:** Class for the efficient drawing of a quadrilateral mesh.

A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
defined by the vertices ::

           (m+1, n) ----------- (m+1, n+1)
              /                   /
             /                 /
            /               /
        (m, n) -------- (m, n+1)

The mesh need not be regular and the polygons need not be convex.

Parameters
----------
coordinates : (M+1, N+1, 2) array-like
    The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
    of vertex (m, n).

antialiased : bool, default: True

shading : {'flat', 'gouraud'}, default: 'flat'

Notes
-----
Unlike other `.Collection`\s, the default *pickradius* of `.QuadMesh` is 0,
i.e. `~.Artist.contains` checks whether the test point is within any of the
mesh quadrilaterals.

## Class: PolyQuadMesh

**Description:** Class for drawing a quadrilateral mesh as individual Polygons.

A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
defined by the vertices ::

           (m+1, n) ----------- (m+1, n+1)
              /                   /
             /                 /
            /               /
        (m, n) -------- (m, n+1)

The mesh need not be regular and the polygons need not be convex.

Parameters
----------
coordinates : (M+1, N+1, 2) array-like
    The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
    of vertex (m, n).

Notes
-----
Unlike `.QuadMesh`, this class will draw each cell as an individual Polygon.
This is significantly slower, but allows for more flexibility when wanting
to add additional properties to the cells, such as hatching.

Another difference from `.QuadMesh` is that if any of the vertices or data
of a cell are masked, that Polygon will **not** be drawn and it won't be in
the list of paths returned.

### Function: __init__(self)

**Description:** Parameters
----------
edgecolors : :mpltype:`color` or list of colors, default: :rc:`patch.edgecolor`
    Edge color for each patch making up the collection. The special
    value 'face' can be passed to make the edgecolor match the
    facecolor.
facecolors : :mpltype:`color` or list of colors, default: :rc:`patch.facecolor`
    Face color for each patch making up the collection.
linewidths : float or list of floats, default: :rc:`patch.linewidth`
    Line width for each patch making up the collection.
linestyles : str or tuple or list thereof, default: 'solid'
    Valid strings are ['solid', 'dashed', 'dashdot', 'dotted', '-',
    '--', '-.', ':']. Dash tuples should be of the form::

        (offset, onoffseq),

    where *onoffseq* is an even length tuple of on and off ink lengths
    in points. For examples, see
    :doc:`/gallery/lines_bars_and_markers/linestyles`.
capstyle : `.CapStyle`-like, default: 'butt'
    Style to use for capping lines for all paths in the collection.
    Allowed values are %(CapStyle)s.
joinstyle : `.JoinStyle`-like, default: 'round'
    Style to use for joining lines for all paths in the collection.
    Allowed values are %(JoinStyle)s.
antialiaseds : bool or list of bool, default: :rc:`patch.antialiased`
    Whether each patch in the collection should be drawn with
    antialiasing.
offsets : (float, float) or list thereof, default: (0, 0)
    A vector by which to translate each patch after rendering (default
    is no translation). The translation is performed in screen (pixel)
    coordinates (i.e. after the Artist's transform is applied).
offset_transform : `~.Transform`, default: `.IdentityTransform`
    A single transform which will be applied to each *offsets* vector
    before it is used.
cmap, norm
    Data normalization and colormapping parameters. See
    `.ScalarMappable` for a detailed description.
hatch : str, optional
    Hatching pattern to use in filled paths, if any. Valid strings are
    ['/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*']. See
    :doc:`/gallery/shapes_and_collections/hatch_style_reference` for
    the meaning of each hatch type.
pickradius : float, default: 5.0
    If ``pickradius <= 0``, then `.Collection.contains` will return
    ``True`` whenever the test point is inside of one of the polygons
    formed by the control points of a Path in the Collection. On the
    other hand, if it is greater than 0, then we instead check if the
    test point is contained in a stroke of width ``2*pickradius``
    following any of the Paths in the Collection.
urls : list of str, default: None
    A URL for each patch to link to once drawn. Currently only works
    for the SVG backend. See :doc:`/gallery/misc/hyperlinks_sgskip` for
    examples.
zorder : float, default: 1
    The drawing order, shared by all Patches in the Collection. See
    :doc:`/gallery/misc/zorder_demo` for all defaults and examples.
**kwargs
    Remaining keyword arguments will be used to set properties as
    ``Collection.set_{key}(val)`` for each key-value pair in *kwargs*.

### Function: get_paths(self)

### Function: set_paths(self, paths)

### Function: get_transforms(self)

### Function: get_offset_transform(self)

**Description:** Return the `.Transform` instance used by this artist offset.

### Function: set_offset_transform(self, offset_transform)

**Description:** Set the artist offset transform.

Parameters
----------
offset_transform : `.Transform`

### Function: get_datalim(self, transData)

### Function: get_window_extent(self, renderer)

### Function: _prepare_points(self)

### Function: draw(self, renderer)

### Function: set_pickradius(self, pickradius)

**Description:** Set the pick radius used for containment tests.

Parameters
----------
pickradius : float
    Pick radius, in points.

### Function: get_pickradius(self)

### Function: contains(self, mouseevent)

**Description:** Test whether the mouse event occurred in the collection.

Returns ``bool, dict(ind=itemlist)``, where every item in itemlist
contains the event.

### Function: set_urls(self, urls)

**Description:** Parameters
----------
urls : list of str or None

Notes
-----
URLs are currently only implemented by the SVG backend. They are
ignored by all other backends.

### Function: get_urls(self)

**Description:** Return a list of URLs, one for each element of the collection.

The list contains *None* for elements without a URL. See
:doc:`/gallery/misc/hyperlinks_sgskip` for an example.

### Function: set_hatch(self, hatch)

**Description:** Set the hatching pattern

*hatch* can be one of::

  /   - diagonal hatching
  \   - back diagonal
  |   - vertical
  -   - horizontal
  +   - crossed
  x   - crossed diagonal
  o   - small circle
  O   - large circle
  .   - dots
  *   - stars

Letters can be combined, in which case all the specified
hatchings are done.  If same letter repeats, it increases the
density of hatching of that pattern.

Unlike other properties such as linewidth and colors, hatching
can only be specified for the collection as a whole, not separately
for each member.

Parameters
----------
hatch : {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}

### Function: get_hatch(self)

**Description:** Return the current hatching pattern.

### Function: set_hatch_linewidth(self, lw)

**Description:** Set the hatch linewidth.

### Function: get_hatch_linewidth(self)

**Description:** Return the hatch linewidth.

### Function: set_offsets(self, offsets)

**Description:** Set the offsets for the collection.

Parameters
----------
offsets : (N, 2) or (2,) array-like

### Function: get_offsets(self)

**Description:** Return the offsets for the collection.

### Function: _get_default_linewidth(self)

### Function: set_linewidth(self, lw)

**Description:** Set the linewidth(s) for the collection.  *lw* can be a scalar
or a sequence; if it is a sequence the patches will cycle
through the sequence

Parameters
----------
lw : float or list of floats

### Function: set_linestyle(self, ls)

**Description:** Set the linestyle(s) for the collection.

===========================   =================
linestyle                     description
===========================   =================
``'-'`` or ``'solid'``        solid line
``'--'`` or  ``'dashed'``     dashed line
``'-.'`` or  ``'dashdot'``    dash-dotted line
``':'`` or ``'dotted'``       dotted line
===========================   =================

Alternatively a dash tuple of the following form can be provided::

    (offset, onoffseq),

where ``onoffseq`` is an even length tuple of on and off ink in points.

Parameters
----------
ls : str or tuple or list thereof
    Valid values for individual linestyles include {'-', '--', '-.',
    ':', '', (offset, on-off-seq)}. See `.Line2D.set_linestyle` for a
    complete description.

### Function: set_capstyle(self, cs)

**Description:** Set the `.CapStyle` for the collection (for all its elements).

Parameters
----------
cs : `.CapStyle` or %(CapStyle)s

### Function: get_capstyle(self)

**Description:** Return the cap style for the collection (for all its elements).

Returns
-------
%(CapStyle)s or None

### Function: set_joinstyle(self, js)

**Description:** Set the `.JoinStyle` for the collection (for all its elements).

Parameters
----------
js : `.JoinStyle` or %(JoinStyle)s

### Function: get_joinstyle(self)

**Description:** Return the join style for the collection (for all its elements).

Returns
-------
%(JoinStyle)s or None

### Function: _bcast_lwls(linewidths, dashes)

**Description:** Internal helper function to broadcast + scale ls/lw

In the collection drawing code, the linewidth and linestyle are cycled
through as circular buffers (via ``v[i % len(v)]``).  Thus, if we are
going to scale the dash pattern at set time (not draw time) we need to
do the broadcasting now and expand both lists to be the same length.

Parameters
----------
linewidths : list
    line widths of collection
dashes : list
    dash specification (offset, (dash pattern tuple))

Returns
-------
linewidths, dashes : list
    Will be the same length, dashes are scaled by paired linewidth

### Function: get_antialiased(self)

**Description:** Get the antialiasing state for rendering.

Returns
-------
array of bools

### Function: set_antialiased(self, aa)

**Description:** Set the antialiasing state for rendering.

Parameters
----------
aa : bool or list of bools

### Function: _get_default_antialiased(self)

### Function: set_color(self, c)

**Description:** Set both the edgecolor and the facecolor.

Parameters
----------
c : :mpltype:`color` or list of RGBA tuples

See Also
--------
Collection.set_facecolor, Collection.set_edgecolor
    For setting the edge or face color individually.

### Function: _get_default_facecolor(self)

### Function: _set_facecolor(self, c)

### Function: set_facecolor(self, c)

**Description:** Set the facecolor(s) of the collection. *c* can be a color (all patches
have same color), or a sequence of colors; if it is a sequence the
patches will cycle through the sequence.

If *c* is 'none', the patch will not be filled.

Parameters
----------
c : :mpltype:`color` or list of :mpltype:`color`

### Function: get_facecolor(self)

### Function: get_edgecolor(self)

### Function: _get_default_edgecolor(self)

### Function: _set_edgecolor(self, c)

### Function: set_edgecolor(self, c)

**Description:** Set the edgecolor(s) of the collection.

Parameters
----------
c : :mpltype:`color` or list of :mpltype:`color` or 'face'
    The collection edgecolor(s).  If a sequence, the patches cycle
    through it.  If 'face', match the facecolor.

### Function: set_alpha(self, alpha)

**Description:** Set the transparency of the collection.

Parameters
----------
alpha : float or array of float or None
    If not None, *alpha* values must be between 0 and 1, inclusive.
    If an array is provided, its length must match the number of
    elements in the collection.  Masked values and nans are not
    supported.

### Function: get_linewidth(self)

### Function: get_linestyle(self)

### Function: _set_mappable_flags(self)

**Description:** Determine whether edges and/or faces are color-mapped.

This is a helper for update_scalarmappable.
It sets Boolean flags '_edge_is_mapped' and '_face_is_mapped'.

Returns
-------
mapping_change : bool
    True if either flag is True, or if a flag has changed.

### Function: update_scalarmappable(self)

**Description:** Update colors from the scalar mappable array, if any.

Assign colors to edges and faces based on the array and/or
colors that were directly set, as appropriate.

### Function: get_fill(self)

**Description:** Return whether face is colored.

### Function: update_from(self, other)

**Description:** Copy properties from other to self.

### Function: get_sizes(self)

**Description:** Return the sizes ('areas') of the elements in the collection.

Returns
-------
array
    The 'area' of each element.

### Function: set_sizes(self, sizes, dpi)

**Description:** Set the sizes of each member of the collection.

Parameters
----------
sizes : `numpy.ndarray` or None
    The size to set for each element of the collection.  The
    value is the 'area' of the element.
dpi : float, default: 72
    The dpi of the canvas.

### Function: draw(self, renderer)

### Function: __init__(self, paths, sizes)

**Description:** Parameters
----------
paths : list of `.path.Path`
    The paths that will make up the `.Collection`.
sizes : array-like
    The factor by which to scale each drawn `~.path.Path`. One unit
    squared in the Path's data space is scaled to be ``sizes**2``
    points when rendered.
**kwargs
    Forwarded to `.Collection`.

### Function: get_paths(self)

### Function: legend_elements(self, prop, num, fmt, func)

**Description:** Create legend handles and labels for a PathCollection.

Each legend handle is a `.Line2D` representing the Path that was drawn,
and each label is a string that represents the Path.

This is useful for obtaining a legend for a `~.Axes.scatter` plot;
e.g.::

    scatter = plt.scatter([1, 2, 3],  [4, 5, 6],  c=[7, 2, 3], num=None)
    plt.legend(*scatter.legend_elements())

creates three legend elements, one for each color with the numerical
values passed to *c* as the labels.

Also see the :ref:`automatedlegendcreation` example.

Parameters
----------
prop : {"colors", "sizes"}, default: "colors"
    If "colors", the legend handles will show the different colors of
    the collection. If "sizes", the legend will show the different
    sizes. To set both, use *kwargs* to directly edit the `.Line2D`
    properties.
num : int, None, "auto" (default), array-like, or `~.ticker.Locator`
    Target number of elements to create.
    If None, use all unique elements of the mappable array. If an
    integer, target to use *num* elements in the normed range.
    If *"auto"*, try to determine which option better suits the nature
    of the data.
    The number of created elements may slightly deviate from *num* due
    to a `~.ticker.Locator` being used to find useful locations.
    If a list or array, use exactly those elements for the legend.
    Finally, a `~.ticker.Locator` can be provided.
fmt : str, `~matplotlib.ticker.Formatter`, or None (default)
    The format or formatter to use for the labels. If a string must be
    a valid input for a `.StrMethodFormatter`. If None (the default),
    use a `.ScalarFormatter`.
func : function, default: ``lambda x: x``
    Function to calculate the labels.  Often the size (or color)
    argument to `~.Axes.scatter` will have been pre-processed by the
    user using a function ``s = f(x)`` to make the markers visible;
    e.g. ``size = np.log10(x)``.  Providing the inverse of this
    function here allows that pre-processing to be inverted, so that
    the legend labels have the correct values; e.g. ``func = lambda
    x: 10**x``.
**kwargs
    Allowed keyword arguments are *color* and *size*. E.g. it may be
    useful to set the color of the markers if *prop="sizes"* is used;
    similarly to set the size of the markers if *prop="colors"* is
    used. Any further parameters are passed onto the `.Line2D`
    instance. This may be useful to e.g. specify a different
    *markeredgecolor* or *alpha* for the legend handles.

Returns
-------
handles : list of `.Line2D`
    Visual representation of each element of the legend.
labels : list of str
    The string labels for elements of the legend.

### Function: __init__(self, verts, sizes)

**Description:** Parameters
----------
verts : list of array-like
    The sequence of polygons [*verts0*, *verts1*, ...] where each
    element *verts_i* defines the vertices of polygon *i* as a 2D
    array-like of shape (M, 2).
sizes : array-like, default: None
    Squared scaling factors for the polygons. The coordinates of each
    polygon *verts_i* are multiplied by the square-root of the
    corresponding entry in *sizes* (i.e., *sizes* specify the scaling
    of areas). The scaling is applied before the Artist master
    transform.
closed : bool, default: True
    Whether the polygon should be closed by adding a CLOSEPOLY
    connection at the end.
**kwargs
    Forwarded to `.Collection`.

### Function: set_verts(self, verts, closed)

**Description:** Set the vertices of the polygons.

Parameters
----------
verts : list of array-like
    The sequence of polygons [*verts0*, *verts1*, ...] where each
    element *verts_i* defines the vertices of polygon *i* as a 2D
    array-like of shape (M, 2).
closed : bool, default: True
    Whether the polygon should be closed by adding a CLOSEPOLY
    connection at the end.

### Function: set_verts_and_codes(self, verts, codes)

**Description:** Initialize vertices with path codes.

### Function: __init__(self, t_direction, t, f1, f2)

**Description:** Parameters
----------
t_direction : {{'x', 'y'}}
    The axes on which the variable lies.

    - 'x': the curves are ``(t, f1)`` and ``(t, f2)``.
    - 'y': the curves are ``(f1, t)`` and ``(f2, t)``.

t : array-like
    The ``t_direction`` coordinates of the nodes defining the curves.

f1 : array-like or float
    The other coordinates of the nodes defining the first curve.

f2 : array-like or float
    The other coordinates of the nodes defining the second curve.

where : array-like of bool, optional
    Define *where* to exclude some {dir} regions from being filled.
    The filled regions are defined by the coordinates ``t[where]``.
    More precisely, fill between ``t[i]`` and ``t[i+1]`` if
    ``where[i] and where[i+1]``.  Note that this definition implies
    that an isolated *True* value between two *False* values in *where*
    will not result in filling.  Both sides of the *True* position
    remain unfilled due to the adjacent *False* values.

interpolate : bool, default: False
    This option is only relevant if *where* is used and the two curves
    are crossing each other.

    Semantically, *where* is often used for *f1* > *f2* or
    similar.  By default, the nodes of the polygon defining the filled
    region will only be placed at the positions in the *t* array.
    Such a polygon cannot describe the above semantics close to the
    intersection.  The t-sections containing the intersection are
    simply clipped.

    Setting *interpolate* to *True* will calculate the actual
    intersection point and extend the filled region up to this point.

step : {{'pre', 'post', 'mid'}}, optional
    Define *step* if the filling should be a step function,
    i.e. constant in between *t*.  The value determines where the
    step will occur:

    - 'pre': The f value is continued constantly to the left from
      every *t* position, i.e. the interval ``(t[i-1], t[i]]`` has the
      value ``f[i]``.
    - 'post': The y value is continued constantly to the right from
      every *x* position, i.e. the interval ``[t[i], t[i+1])`` has the
      value ``f[i]``.
    - 'mid': Steps occur half-way between the *t* positions.

**kwargs
    Forwarded to `.PolyCollection`.

See Also
--------
.Axes.fill_between, .Axes.fill_betweenx

### Function: _f_dir_from_t(t_direction)

**Description:** The direction that is other than `t_direction`.

### Function: _f_direction(self)

**Description:** The direction that is other than `self.t_direction`.

### Function: set_data(self, t, f1, f2)

**Description:** Set new values for the two bounding curves.

Parameters
----------
t : array-like
    The ``self.t_direction`` coordinates of the nodes defining the curves.

f1 : array-like or float
    The other coordinates of the nodes defining the first curve.

f2 : array-like or float
    The other coordinates of the nodes defining the second curve.

where : array-like of bool, optional
    Define *where* to exclude some {dir} regions from being filled.
    The filled regions are defined by the coordinates ``t[where]``.
    More precisely, fill between ``t[i]`` and ``t[i+1]`` if
    ``where[i] and where[i+1]``.  Note that this definition implies
    that an isolated *True* value between two *False* values in *where*
    will not result in filling.  Both sides of the *True* position
    remain unfilled due to the adjacent *False* values.

See Also
--------
.PolyCollection.set_verts, .Line2D.set_data

### Function: get_datalim(self, transData)

**Description:** Calculate the data limits and return them as a `.Bbox`.

### Function: _make_verts(self, t, f1, f2, where)

**Description:** Make verts that can be forwarded to `.PolyCollection`.

### Function: _get_data_mask(self, t, f1, f2, where)

**Description:** Return a bool array, with True at all points that should eventually be rendered.

The array is True at a point if none of the data inputs
*t*, *f1*, *f2* is masked and if the input *where* is true at that point.

### Function: _validate_shapes(t_dir, f_dir, t, f1, f2)

**Description:** Validate that t, f1 and f2 are 1-dimensional and have the same length.

### Function: _make_verts_for_region(self, t, f1, f2, idx0, idx1)

**Description:** Make ``verts`` for a contiguous region between ``idx0`` and ``idx1``, taking
into account ``step`` and ``interpolate``.

### Function: _get_interpolating_points(cls, t, f1, f2, idx)

**Description:** Calculate interpolating points.

### Function: _get_diff_root(x, xp, fp)

**Description:** Calculate diff root.

### Function: _fix_pts_xy_order(self, pts)

**Description:** Fix pts calculation results with `self.t_direction`.

In the workflow, it is assumed that `self.t_direction` is 'x'. If this
is not true, we need to exchange the coordinates.

### Function: __init__(self, numsides)

**Description:** Parameters
----------
numsides : int
    The number of sides of the polygon.
rotation : float
    The rotation of the polygon in radians.
sizes : tuple of float
    The area of the circle circumscribing the polygon in points^2.
**kwargs
    Forwarded to `.Collection`.

Examples
--------
See :doc:`/gallery/event_handling/lasso_demo` for a complete example::

    offsets = np.random.rand(20, 2)
    facecolors = [cm.jet(x) for x in np.random.rand(20)]

    collection = RegularPolyCollection(
        numsides=5, # a pentagon
        rotation=0, sizes=(50,),
        facecolors=facecolors,
        edgecolors=("black",),
        linewidths=(1,),
        offsets=offsets,
        offset_transform=ax.transData,
        )

### Function: get_numsides(self)

### Function: get_rotation(self)

### Function: draw(self, renderer)

### Function: __init__(self, segments)

**Description:** Parameters
----------
segments : list of (N, 2) array-like
    A sequence ``[line0, line1, ...]`` where each line is a (N, 2)-shape
    array-like containing points::

        line0 = [(x0, y0), (x1, y1), ...]

    Each line can contain a different number of points.
linewidths : float or list of float, default: :rc:`lines.linewidth`
    The width of each line in points.
colors : :mpltype:`color` or list of color, default: :rc:`lines.color`
    A sequence of RGBA tuples (e.g., arbitrary color strings, etc, not
    allowed).
antialiaseds : bool or list of bool, default: :rc:`lines.antialiased`
    Whether to use antialiasing for each line.
zorder : float, default: 2
    zorder of the lines once drawn.

facecolors : :mpltype:`color` or list of :mpltype:`color`, default: 'none'
    When setting *facecolors*, each line is interpreted as a boundary
    for an area, implicitly closing the path from the last point to the
    first point. The enclosed area is filled with *facecolor*.
    In order to manually specify what should count as the "interior" of
    each line, please use `.PathCollection` instead, where the
    "interior" can be specified by appropriate usage of
    `~.path.Path.CLOSEPOLY`.

**kwargs
    Forwarded to `.Collection`.

### Function: set_segments(self, segments)

### Function: get_segments(self)

**Description:** Returns
-------
list
    List of segments in the LineCollection. Each list item contains an
    array of vertices.

### Function: _get_default_linewidth(self)

### Function: _get_default_antialiased(self)

### Function: _get_default_edgecolor(self)

### Function: _get_default_facecolor(self)

### Function: set_alpha(self, alpha)

### Function: set_color(self, c)

**Description:** Set the edgecolor(s) of the LineCollection.

Parameters
----------
c : :mpltype:`color` or list of :mpltype:`color`
    Single color (all lines have same color), or a
    sequence of RGBA tuples; if it is a sequence the lines will
    cycle through the sequence.

### Function: get_color(self)

### Function: set_gapcolor(self, gapcolor)

**Description:** Set a color to fill the gaps in the dashed line style.

.. note::

    Striped lines are created by drawing two interleaved dashed lines.
    There can be overlaps between those two, which may result in
    artifacts when using transparency.

    This functionality is experimental and may change.

Parameters
----------
gapcolor : :mpltype:`color` or list of :mpltype:`color` or None
    The color with which to fill the gaps. If None, the gaps are
    unfilled.

### Function: _set_gapcolor(self, gapcolor)

### Function: get_gapcolor(self)

### Function: _get_inverse_paths_linestyles(self)

**Description:** Returns the path and pattern for the gaps in the non-solid lines.

This path and pattern is the inverse of the path and pattern used to
construct the non-solid lines. For solid lines, we set the inverse path
to nans to prevent drawing an inverse line.

### Function: __init__(self, positions, orientation)

**Description:** Parameters
----------
positions : 1D array-like
    Each value is an event.
orientation : {'horizontal', 'vertical'}, default: 'horizontal'
    The sequence of events is plotted along this direction.
    The marker lines of the single events are along the orthogonal
    direction.
lineoffset : float, default: 0
    The offset of the center of the markers from the origin, in the
    direction orthogonal to *orientation*.
linelength : float, default: 1
    The total height of the marker (i.e. the marker stretches from
    ``lineoffset - linelength/2`` to ``lineoffset + linelength/2``).
linewidth : float or list thereof, default: :rc:`lines.linewidth`
    The line width of the event lines, in points.
color : :mpltype:`color` or list of :mpltype:`color`, default: :rc:`lines.color`
    The color of the event lines.
linestyle : str or tuple or list thereof, default: 'solid'
    Valid strings are ['solid', 'dashed', 'dashdot', 'dotted',
    '-', '--', '-.', ':']. Dash tuples should be of the form::

        (offset, onoffseq),

    where *onoffseq* is an even length tuple of on and off ink
    in points.
antialiased : bool or list thereof, default: :rc:`lines.antialiased`
    Whether to use antialiasing for drawing the lines.
**kwargs
    Forwarded to `.LineCollection`.

Examples
--------
.. plot:: gallery/lines_bars_and_markers/eventcollection_demo.py

### Function: get_positions(self)

**Description:** Return an array containing the floating-point values of the positions.

### Function: set_positions(self, positions)

**Description:** Set the positions of the events.

### Function: add_positions(self, position)

**Description:** Add one or more events at the specified positions.

### Function: is_horizontal(self)

**Description:** True if the eventcollection is horizontal, False if vertical.

### Function: get_orientation(self)

**Description:** Return the orientation of the event line ('horizontal' or 'vertical').

### Function: switch_orientation(self)

**Description:** Switch the orientation of the event line, either from vertical to
horizontal or vice versus.

### Function: set_orientation(self, orientation)

**Description:** Set the orientation of the event line.

Parameters
----------
orientation : {'horizontal', 'vertical'}

### Function: get_linelength(self)

**Description:** Return the length of the lines used to mark each event.

### Function: set_linelength(self, linelength)

**Description:** Set the length of the lines used to mark each event.

### Function: get_lineoffset(self)

**Description:** Return the offset of the lines used to mark each event.

### Function: set_lineoffset(self, lineoffset)

**Description:** Set the offset of the lines used to mark each event.

### Function: get_linewidth(self)

**Description:** Get the width of the lines used to mark each event.

### Function: get_linewidths(self)

### Function: get_color(self)

**Description:** Return the color of the lines used to mark each event.

### Function: __init__(self, sizes)

**Description:** Parameters
----------
sizes : float or array-like
    The area of each circle in points^2.
**kwargs
    Forwarded to `.Collection`.

### Function: __init__(self, widths, heights, angles)

**Description:** Parameters
----------
widths : array-like
    The lengths of the first axes (e.g., major axis lengths).
heights : array-like
    The lengths of second axes.
angles : array-like
    The angles of the first axes, degrees CCW from the x-axis.
units : {'points', 'inches', 'dots', 'width', 'height', 'x', 'y', 'xy'}
    The units in which majors and minors are given; 'width' and
    'height' refer to the dimensions of the axes, while 'x' and 'y'
    refer to the *offsets* data units. 'xy' differs from all others in
    that the angle as plotted varies with the aspect ratio, and equals
    the specified angle only when the aspect ratio is unity.  Hence
    it behaves the same as the `~.patches.Ellipse` with
    ``axes.transData`` as its transform.
**kwargs
    Forwarded to `Collection`.

### Function: _set_transforms(self)

**Description:** Calculate transforms immediately before drawing.

### Function: set_widths(self, widths)

**Description:** Set the lengths of the first axes (e.g., major axis).

### Function: set_heights(self, heights)

**Description:** Set the lengths of second axes (e.g., minor axes).

### Function: set_angles(self, angles)

**Description:** Set the angles of the first axes, degrees CCW from the x-axis.

### Function: get_widths(self)

**Description:** Get the lengths of the first axes (e.g., major axis).

### Function: get_heights(self)

**Description:** Set the lengths of second axes (e.g., minor axes).

### Function: get_angles(self)

**Description:** Get the angles of the first axes, degrees CCW from the x-axis.

### Function: draw(self, renderer)

### Function: __init__(self, patches)

**Description:** Parameters
----------
patches : list of `.Patch`
    A sequence of Patch objects.  This list may include
    a heterogeneous assortment of different patch types.

match_original : bool, default: False
    If True, use the colors and linewidths of the original
    patches.  If False, new colors may be assigned by
    providing the standard collection arguments, facecolor,
    edgecolor, linewidths, norm or cmap.

**kwargs
    All other parameters are forwarded to `.Collection`.

    If any of *edgecolors*, *facecolors*, *linewidths*, *antialiaseds*
    are None, they default to their `.rcParams` patch setting, in
    sequence form.

Notes
-----
The use of `~matplotlib.cm.ScalarMappable` functionality is optional.
If the `~matplotlib.cm.ScalarMappable` matrix ``_A`` has been set (via
a call to `~.ScalarMappable.set_array`), at draw time a call to scalar
mappable will be made to set the face colors.

### Function: set_paths(self, patches)

### Function: __init__(self, triangulation)

### Function: get_paths(self)

### Function: set_paths(self)

### Function: convert_mesh_to_paths(tri)

**Description:** Convert a given mesh into a sequence of `.Path` objects.

This function is primarily of use to implementers of backends that do
not directly support meshes.

### Function: draw(self, renderer)

### Function: __init__(self, coordinates)

### Function: set_array(self, A)

**Description:** Set the data values.

Parameters
----------
A : array-like
    The mesh data. Supported array shapes are:

    - (M, N) or (M*N,): a mesh with scalar data. The values are mapped
      to colors using normalization and a colormap. See parameters
      *norm*, *cmap*, *vmin*, *vmax*.
    - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
    - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),
      i.e. including transparency.

    If the values are provided as a 2D grid, the shape must match the
    coordinates grid. If the values are 1D, they are reshaped to 2D.
    M, N follow from the coordinates grid, where the coordinates grid
    shape is (M, N) for 'gouraud' *shading* and (M+1, N+1) for 'flat'
    shading.

### Function: get_coordinates(self)

**Description:** Return the vertices of the mesh as an (M+1, N+1, 2) array.

M, N are the number of quadrilaterals in the rows / columns of the
mesh, corresponding to (M+1, N+1) vertices.
The last dimension specifies the components (x, y).

### Function: get_edgecolor(self)

### Function: get_facecolor(self)

### Function: _convert_mesh_to_paths(coordinates)

**Description:** Convert a given mesh into a sequence of `.Path` objects.

This function is primarily of use to implementers of backends that do
not directly support quadmeshes.

### Function: _convert_mesh_to_triangles(self, coordinates)

**Description:** Convert a given mesh into a sequence of triangles, each point
with its own color.  The result can be used to construct a call to
`~.RendererBase.draw_gouraud_triangles`.

### Function: __init__(self, coordinates)

### Function: get_paths(self)

### Function: set_paths(self)

### Function: get_datalim(self, transData)

### Function: draw(self, renderer)

### Function: get_cursor_data(self, event)

### Function: __init__(self, coordinates)

### Function: _get_unmasked_polys(self)

**Description:** Get the unmasked regions using the coordinates and array

### Function: _set_unmasked_verts(self)

### Function: get_edgecolor(self)

### Function: get_facecolor(self)

### Function: set_array(self, A)

### Function: determine_facecolor(patch)
