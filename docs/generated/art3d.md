## AI Summary

A file named art3d.py.


### Function: _norm_angle(a)

**Description:** Return the given angle normalized to -180 < *a* <= 180 degrees.

### Function: _norm_text_angle(a)

**Description:** Return the given angle normalized to -90 < *a* <= 90 degrees.

### Function: get_dir_vector(zdir)

**Description:** Return a direction vector.

Parameters
----------
zdir : {'x', 'y', 'z', None, 3-tuple}
    The direction. Possible values are:

    - 'x': equivalent to (1, 0, 0)
    - 'y': equivalent to (0, 1, 0)
    - 'z': equivalent to (0, 0, 1)
    - *None*: equivalent to (0, 0, 0)
    - an iterable (x, y, z) is converted to an array

Returns
-------
x, y, z : array
    The direction vector.

### Function: _viewlim_mask(xs, ys, zs, axes)

**Description:** Return original points with points outside the axes view limits masked.

Parameters
----------
xs, ys, zs : array-like
    The points to mask.
axes : Axes3D
    The axes to use for the view limits.

Returns
-------
xs_masked, ys_masked, zs_masked : np.ma.array
    The masked points.

## Class: Text3D

**Description:** Text object with 3D position and direction.

Parameters
----------
x, y, z : float
    The position of the text.
text : str
    The text string to display.
zdir : {'x', 'y', 'z', None, 3-tuple}
    The direction of the text. See `.get_dir_vector` for a description of
    the values.
axlim_clip : bool, default: False
    Whether to hide text outside the axes view limits.

Other Parameters
----------------
**kwargs
     All other parameters are passed on to `~matplotlib.text.Text`.

### Function: text_2d_to_3d(obj, z, zdir, axlim_clip)

**Description:** Convert a `.Text` to a `.Text3D` object.

Parameters
----------
z : float
    The z-position in 3D space.
zdir : {'x', 'y', 'z', 3-tuple}
    The direction of the text. Default: 'z'.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide text outside the axes view limits.

## Class: Line3D

**Description:** 3D line object.

.. note:: Use `get_data_3d` to obtain the data associated with the line.
        `~.Line2D.get_data`, `~.Line2D.get_xdata`, and `~.Line2D.get_ydata` return
        the x- and y-coordinates of the projected 2D-line, not the x- and y-data of
        the 3D-line. Similarly, use `set_data_3d` to set the data, not
        `~.Line2D.set_data`, `~.Line2D.set_xdata`, and `~.Line2D.set_ydata`.

### Function: line_2d_to_3d(line, zs, zdir, axlim_clip)

**Description:** Convert a `.Line2D` to a `.Line3D` object.

Parameters
----------
zs : float
    The location along the *zdir* axis in 3D space to position the line.
zdir : {'x', 'y', 'z'}
    Plane to plot line orthogonal to. Default: 'z'.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide lines with an endpoint outside the axes view limits.

### Function: _path_to_3d_segment(path, zs, zdir)

**Description:** Convert a path to a 3D segment.

### Function: _paths_to_3d_segments(paths, zs, zdir)

**Description:** Convert paths from a collection object to 3D segments.

### Function: _path_to_3d_segment_with_codes(path, zs, zdir)

**Description:** Convert a path to a 3D segment with path codes.

### Function: _paths_to_3d_segments_with_codes(paths, zs, zdir)

**Description:** Convert paths from a collection object to 3D segments with path codes.

## Class: Collection3D

**Description:** A collection of 3D paths.

### Function: collection_2d_to_3d(col, zs, zdir, axlim_clip)

**Description:** Convert a `.Collection` to a `.Collection3D` object.

## Class: Line3DCollection

**Description:** A collection of 3D lines.

### Function: line_collection_2d_to_3d(col, zs, zdir, axlim_clip)

**Description:** Convert a `.LineCollection` to a `.Line3DCollection` object.

## Class: Patch3D

**Description:** 3D patch object.

## Class: PathPatch3D

**Description:** 3D PathPatch object.

### Function: _get_patch_verts(patch)

**Description:** Return a list of vertices for the path of a patch.

### Function: patch_2d_to_3d(patch, z, zdir, axlim_clip)

**Description:** Convert a `.Patch` to a `.Patch3D` object.

### Function: pathpatch_2d_to_3d(pathpatch, z, zdir)

**Description:** Convert a `.PathPatch` to a `.PathPatch3D` object.

## Class: Patch3DCollection

**Description:** A collection of 3D patches.

## Class: Path3DCollection

**Description:** A collection of 3D paths.

### Function: patch_collection_2d_to_3d(col, zs, zdir, depthshade, axlim_clip)

**Description:** Convert a `.PatchCollection` into a `.Patch3DCollection` object
(or a `.PathCollection` into a `.Path3DCollection` object).

Parameters
----------
col : `~matplotlib.collections.PatchCollection` or `~matplotlib.collections.PathCollection`
    The collection to convert.
zs : float or array of floats
    The location or locations to place the patches in the collection along
    the *zdir* axis. Default: 0.
zdir : {'x', 'y', 'z'}
    The axis in which to place the patches. Default: "z".
    See `.get_dir_vector` for a description of the values.
depthshade : bool, default: True
    Whether to shade the patches to give a sense of depth.
axlim_clip : bool, default: False
    Whether to hide patches with a vertex outside the axes view limits.

## Class: Poly3DCollection

**Description:** A collection of 3D polygons.

.. note::
    **Filling of 3D polygons**

    There is no simple definition of the enclosed surface of a 3D polygon
    unless the polygon is planar.

    In practice, Matplotlib fills the 2D projection of the polygon. This
    gives a correct filling appearance only for planar polygons. For all
    other polygons, you'll find orientations in which the edges of the
    polygon intersect in the projection. This will lead to an incorrect
    visualization of the 3D area.

    If you need filled areas, it is recommended to create them via
    `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`, which creates a
    triangulation and thus generates consistent surfaces.

### Function: poly_collection_2d_to_3d(col, zs, zdir, axlim_clip)

**Description:** Convert a `.PolyCollection` into a `.Poly3DCollection` object.

Parameters
----------
col : `~matplotlib.collections.PolyCollection`
    The collection to convert.
zs : float or array of floats
    The location or locations to place the polygons in the collection along
    the *zdir* axis. Default: 0.
zdir : {'x', 'y', 'z'}
    The axis in which to place the patches. Default: 'z'.
    See `.get_dir_vector` for a description of the values.

### Function: juggle_axes(xs, ys, zs, zdir)

**Description:** Reorder coordinates so that 2D *xs*, *ys* can be plotted in the plane
orthogonal to *zdir*. *zdir* is normally 'x', 'y' or 'z'. However, if
*zdir* starts with a '-' it is interpreted as a compensation for
`rotate_axes`.

### Function: rotate_axes(xs, ys, zs, zdir)

**Description:** Reorder coordinates so that the axes are rotated with *zdir* along
the original z axis. Prepending the axis with a '-' does the
inverse transform, so *zdir* can be 'x', '-x', 'y', '-y', 'z' or '-z'.

### Function: _zalpha(colors, zs)

**Description:** Modify the alphas of the color list according to depth.

### Function: _all_points_on_plane(xs, ys, zs, atol)

**Description:** Check if all points are on the same plane. Note that NaN values are
ignored.

Parameters
----------
xs, ys, zs : array-like
    The x, y, and z coordinates of the points.
atol : float, default: 1e-8
    The tolerance for the equality check.

### Function: _generate_normals(polygons)

**Description:** Compute the normals of a list of polygons, one normal per polygon.

Normals point towards the viewer for a face with its vertices in
counterclockwise order, following the right hand rule.

Uses three points equally spaced around the polygon. This method assumes
that the points are in a plane. Otherwise, more than one shade is required,
which is not supported.

Parameters
----------
polygons : list of (M_i, 3) array-like, or (..., M, 3) array-like
    A sequence of polygons to compute normals for, which can have
    varying numbers of vertices. If the polygons all have the same
    number of vertices and array is passed, then the operation will
    be vectorized.

Returns
-------
normals : (..., 3) array
    A normal vector estimated for the polygon.

### Function: _shade_colors(color, normals, lightsource)

**Description:** Shade *color* using normal vectors given by *normals*,
assuming a *lightsource* (using default position if not given).
*color* can also be an array of the same length as *normals*.

### Function: __init__(self, x, y, z, text, zdir, axlim_clip)

### Function: get_position_3d(self)

**Description:** Return the (x, y, z) position of the text.

### Function: set_position_3d(self, xyz, zdir)

**Description:** Set the (*x*, *y*, *z*) position of the text.

Parameters
----------
xyz : (float, float, float)
    The position in 3D space.
zdir : {'x', 'y', 'z', None, 3-tuple}
    The direction of the text. If unspecified, the *zdir* will not be
    changed. See `.get_dir_vector` for a description of the values.

### Function: set_z(self, z)

**Description:** Set the *z* position of the text.

Parameters
----------
z : float

### Function: set_3d_properties(self, z, zdir, axlim_clip)

**Description:** Set the *z* position and direction of the text.

Parameters
----------
z : float
    The z-position in 3D space.
zdir : {'x', 'y', 'z', 3-tuple}
    The direction of the text. Default: 'z'.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide text outside the axes view limits.

### Function: draw(self, renderer)

### Function: get_tightbbox(self, renderer)

### Function: __init__(self, xs, ys, zs)

**Description:** Parameters
----------
xs : array-like
    The x-data to be plotted.
ys : array-like
    The y-data to be plotted.
zs : array-like
    The z-data to be plotted.
*args, **kwargs
    Additional arguments are passed to `~matplotlib.lines.Line2D`.

### Function: set_3d_properties(self, zs, zdir, axlim_clip)

**Description:** Set the *z* position and direction of the line.

Parameters
----------
zs : float or array of floats
    The location along the *zdir* axis in 3D space to position the
    line.
zdir : {'x', 'y', 'z'}
    Plane to plot line orthogonal to. Default: 'z'.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide lines with an endpoint outside the axes view limits.

### Function: set_data_3d(self)

**Description:** Set the x, y and z data

Parameters
----------
x : array-like
    The x-data to be plotted.
y : array-like
    The y-data to be plotted.
z : array-like
    The z-data to be plotted.

Notes
-----
Accepts x, y, z arguments or a single array-like (x, y, z)

### Function: get_data_3d(self)

**Description:** Get the current data

Returns
-------
verts3d : length-3 tuple or array-like
    The current data as a tuple or array-like.

### Function: draw(self, renderer)

### Function: do_3d_projection(self)

**Description:** Project the points according to renderer matrix.

### Function: __init__(self, lines, axlim_clip)

### Function: set_sort_zpos(self, val)

**Description:** Set the position to use for z-sorting.

### Function: set_segments(self, segments)

**Description:** Set 3D segments.

### Function: do_3d_projection(self)

**Description:** Project the points according to renderer matrix.

### Function: __init__(self)

**Description:** Parameters
----------
verts :
zs : float
    The location along the *zdir* axis in 3D space to position the
    patch.
zdir : {'x', 'y', 'z'}
    Plane to plot patch orthogonal to. Default: 'z'.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide patches with a vertex outside the axes view limits.

### Function: set_3d_properties(self, verts, zs, zdir, axlim_clip)

**Description:** Set the *z* position and direction of the patch.

Parameters
----------
verts :
zs : float
    The location along the *zdir* axis in 3D space to position the
    patch.
zdir : {'x', 'y', 'z'}
    Plane to plot patch orthogonal to. Default: 'z'.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide patches with a vertex outside the axes view limits.

### Function: get_path(self)

### Function: do_3d_projection(self)

### Function: __init__(self, path)

**Description:** Parameters
----------
path :
zs : float
    The location along the *zdir* axis in 3D space to position the
    path patch.
zdir : {'x', 'y', 'z', 3-tuple}
    Plane to plot path patch orthogonal to. Default: 'z'.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide path patches with a point outside the axes view limits.

### Function: set_3d_properties(self, path, zs, zdir, axlim_clip)

**Description:** Set the *z* position and direction of the path patch.

Parameters
----------
path :
zs : float
    The location along the *zdir* axis in 3D space to position the
    path patch.
zdir : {'x', 'y', 'z', 3-tuple}
    Plane to plot path patch orthogonal to. Default: 'z'.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide path patches with a point outside the axes view limits.

### Function: do_3d_projection(self)

### Function: __init__(self)

**Description:** Create a collection of flat 3D patches with its normal vector
pointed in *zdir* direction, and located at *zs* on the *zdir*
axis. 'zs' can be a scalar or an array-like of the same length as
the number of patches in the collection.

Constructor arguments are the same as for
:class:`~matplotlib.collections.PatchCollection`. In addition,
keywords *zs=0* and *zdir='z'* are available.

Also, the keyword argument *depthshade* is available to indicate
whether to shade the patches in order to give the appearance of depth
(default is *True*). This is typically desired in scatter plots.

### Function: get_depthshade(self)

### Function: set_depthshade(self, depthshade)

**Description:** Set whether depth shading is performed on collection members.

Parameters
----------
depthshade : bool
    Whether to shade the patches in order to give the appearance of
    depth.

### Function: set_sort_zpos(self, val)

**Description:** Set the position to use for z-sorting.

### Function: set_3d_properties(self, zs, zdir, axlim_clip)

**Description:** Set the *z* positions and direction of the patches.

Parameters
----------
zs : float or array of floats
    The location or locations to place the patches in the collection
    along the *zdir* axis.
zdir : {'x', 'y', 'z'}
    Plane to plot patches orthogonal to.
    All patches must have the same direction.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide patches with a vertex outside the axes view limits.

### Function: do_3d_projection(self)

### Function: _maybe_depth_shade_and_sort_colors(self, color_array)

### Function: get_facecolor(self)

### Function: get_edgecolor(self)

### Function: __init__(self)

**Description:** Create a collection of flat 3D paths with its normal vector
pointed in *zdir* direction, and located at *zs* on the *zdir*
axis. 'zs' can be a scalar or an array-like of the same length as
the number of paths in the collection.

Constructor arguments are the same as for
:class:`~matplotlib.collections.PathCollection`. In addition,
keywords *zs=0* and *zdir='z'* are available.

Also, the keyword argument *depthshade* is available to indicate
whether to shade the patches in order to give the appearance of depth
(default is *True*). This is typically desired in scatter plots.

### Function: draw(self, renderer)

### Function: set_sort_zpos(self, val)

**Description:** Set the position to use for z-sorting.

### Function: set_3d_properties(self, zs, zdir, axlim_clip)

**Description:** Set the *z* positions and direction of the paths.

Parameters
----------
zs : float or array of floats
    The location or locations to place the paths in the collection
    along the *zdir* axis.
zdir : {'x', 'y', 'z'}
    Plane to plot paths orthogonal to.
    All paths must have the same direction.
    See `.get_dir_vector` for a description of the values.
axlim_clip : bool, default: False
    Whether to hide paths with a vertex outside the axes view limits.

### Function: set_sizes(self, sizes, dpi)

### Function: set_linewidth(self, lw)

### Function: get_depthshade(self)

### Function: set_depthshade(self, depthshade)

**Description:** Set whether depth shading is performed on collection members.

Parameters
----------
depthshade : bool
    Whether to shade the patches in order to give the appearance of
    depth.

### Function: do_3d_projection(self)

### Function: _use_zordered_offset(self)

### Function: _maybe_depth_shade_and_sort_colors(self, color_array)

### Function: get_facecolor(self)

### Function: get_edgecolor(self)

### Function: __init__(self, verts)

**Description:** Parameters
----------
verts : list of (N, 3) array-like
    The sequence of polygons [*verts0*, *verts1*, ...] where each
    element *verts_i* defines the vertices of polygon *i* as a 2D
    array-like of shape (N, 3).
zsort : {'average', 'min', 'max'}, default: 'average'
    The calculation method for the z-order.
    See `~.Poly3DCollection.set_zsort` for details.
shade : bool, default: False
    Whether to shade *facecolors* and *edgecolors*. When activating
    *shade*, *facecolors* and/or *edgecolors* must be provided.

    .. versionadded:: 3.7

lightsource : `~matplotlib.colors.LightSource`, optional
    The lightsource to use when *shade* is True.

    .. versionadded:: 3.7

axlim_clip : bool, default: False
    Whether to hide polygons with a vertex outside the view limits.

*args, **kwargs
    All other parameters are forwarded to `.PolyCollection`.

Notes
-----
Note that this class does a bit of magic with the _facecolors
and _edgecolors properties.

### Function: set_zsort(self, zsort)

**Description:** Set the calculation method for the z-order.

Parameters
----------
zsort : {'average', 'min', 'max'}
    The function applied on the z-coordinates of the vertices in the
    viewer's coordinate system, to determine the z-order.

### Function: get_vector(self, segments3d)

### Function: _get_vector(self, segments3d)

**Description:** Optimize points for projection.

### Function: set_verts(self, verts, closed)

**Description:** Set 3D vertices.

Parameters
----------
verts : list of (N, 3) array-like
    The sequence of polygons [*verts0*, *verts1*, ...] where each
    element *verts_i* defines the vertices of polygon *i* as a 2D
    array-like of shape (N, 3).
closed : bool, default: True
    Whether the polygon should be closed by adding a CLOSEPOLY
    connection at the end.

### Function: set_verts_and_codes(self, verts, codes)

**Description:** Set 3D vertices with path codes.

### Function: set_3d_properties(self, axlim_clip)

### Function: set_sort_zpos(self, val)

**Description:** Set the position to use for z-sorting.

### Function: do_3d_projection(self)

**Description:** Perform the 3D projection for this object.

### Function: set_facecolor(self, colors)

### Function: set_edgecolor(self, colors)

### Function: set_alpha(self, alpha)

### Function: get_facecolor(self)

### Function: get_edgecolor(self)

### Function: norm(x)
