## AI Summary

A file named transforms.py.


### Function: _make_str_method()

**Description:** Generate a ``__str__`` method for a `.Transform` subclass.

After ::

    class T:
        __str__ = _make_str_method("attr", key="other")

``str(T(...))`` will be

.. code-block:: text

    {type(T).__name__}(
        {self.attr},
        key={self.other})

## Class: TransformNode

**Description:** The base class for anything that participates in the transform tree
and needs to invalidate its parents or be invalidated.  This includes
classes that are not really transforms, such as bounding boxes, since some
transforms depend on bounding boxes to compute their values.

## Class: BboxBase

**Description:** The base class of all bounding boxes.

This class is immutable; `Bbox` is a mutable subclass.

The canonical representation is as two points, with no
restrictions on their ordering.  Convenience properties are
provided to get the left, bottom, right and top edges and width
and height, but these are not stored explicitly.

## Class: Bbox

**Description:** A mutable bounding box.

Examples
--------
**Create from known bounds**

The default constructor takes the boundary "points" ``[[xmin, ymin],
[xmax, ymax]]``.

    >>> Bbox([[1, 1], [3, 7]])
    Bbox([[1.0, 1.0], [3.0, 7.0]])

Alternatively, a Bbox can be created from the flattened points array, the
so-called "extents" ``(xmin, ymin, xmax, ymax)``

    >>> Bbox.from_extents(1, 1, 3, 7)
    Bbox([[1.0, 1.0], [3.0, 7.0]])

or from the "bounds" ``(xmin, ymin, width, height)``.

    >>> Bbox.from_bounds(1, 1, 2, 6)
    Bbox([[1.0, 1.0], [3.0, 7.0]])

**Create from collections of points**

The "empty" object for accumulating Bboxs is the null bbox, which is a
stand-in for the empty set.

    >>> Bbox.null()
    Bbox([[inf, inf], [-inf, -inf]])

Adding points to the null bbox will give you the bbox of those points.

    >>> box = Bbox.null()
    >>> box.update_from_data_xy([[1, 1]])
    >>> box
    Bbox([[1.0, 1.0], [1.0, 1.0]])
    >>> box.update_from_data_xy([[2, 3], [3, 2]], ignore=False)
    >>> box
    Bbox([[1.0, 1.0], [3.0, 3.0]])

Setting ``ignore=True`` is equivalent to starting over from a null bbox.

    >>> box.update_from_data_xy([[1, 1]], ignore=True)
    >>> box
    Bbox([[1.0, 1.0], [1.0, 1.0]])

.. warning::

    It is recommended to always specify ``ignore`` explicitly.  If not, the
    default value of ``ignore`` can be changed at any time by code with
    access to your Bbox, for example using the method `~.Bbox.ignore`.

**Properties of the ``null`` bbox**

.. note::

    The current behavior of `Bbox.null()` may be surprising as it does
    not have all of the properties of the "empty set", and as such does
    not behave like a "zero" object in the mathematical sense. We may
    change that in the future (with a deprecation period).

The null bbox is the identity for intersections

    >>> Bbox.intersection(Bbox([[1, 1], [3, 7]]), Bbox.null())
    Bbox([[1.0, 1.0], [3.0, 7.0]])

except with itself, where it returns the full space.

    >>> Bbox.intersection(Bbox.null(), Bbox.null())
    Bbox([[-inf, -inf], [inf, inf]])

A union containing null will always return the full space (not the other
set!)

    >>> Bbox.union([Bbox([[0, 0], [0, 0]]), Bbox.null()])
    Bbox([[-inf, -inf], [inf, inf]])

## Class: TransformedBbox

**Description:** A `Bbox` that is automatically transformed by a given
transform.  When either the child bounding box or transform
changes, the bounds of this bbox will update accordingly.

## Class: LockableBbox

**Description:** A `Bbox` where some elements may be locked at certain values.

When the child bounding box changes, the bounds of this bbox will update
accordingly with the exception of the locked elements.

## Class: Transform

**Description:** The base class of all `TransformNode` instances that
actually perform a transformation.

All non-affine transformations should be subclasses of this class.
New affine transformations should be subclasses of `Affine2D`.

Subclasses of this class should override the following members (at
minimum):

- :attr:`input_dims`
- :attr:`output_dims`
- :meth:`transform`
- :meth:`inverted` (if an inverse exists)

The following attributes may be overridden if the default is unsuitable:

- :attr:`is_separable` (defaults to True for 1D -> 1D transforms, False
  otherwise)
- :attr:`has_inverse` (defaults to True if :meth:`inverted` is overridden,
  False otherwise)

If the transform needs to do something non-standard with
`matplotlib.path.Path` objects, such as adding curves
where there were once line segments, it should override:

- :meth:`transform_path`

## Class: TransformWrapper

**Description:** A helper class that holds a single child transform and acts
equivalently to it.

This is useful if a node of the transform tree must be replaced at
run time with a transform of a different type.  This class allows
that replacement to correctly trigger invalidation.

`TransformWrapper` instances must have the same input and output dimensions
during their entire lifetime, so the child transform may only be replaced
with another child transform of the same dimensions.

## Class: AffineBase

**Description:** The base class of all affine transformations of any number of dimensions.

## Class: Affine2DBase

**Description:** The base class of all 2D affine transformations.

2D affine transformations are performed using a 3x3 numpy array::

    a c e
    b d f
    0 0 1

This class provides the read-only interface.  For a mutable 2D
affine transformation, use `Affine2D`.

Subclasses of this class will generally only need to override a
constructor and `~.Transform.get_matrix` that generates a custom 3x3 matrix.

## Class: Affine2D

**Description:** A mutable 2D affine transformation.

## Class: IdentityTransform

**Description:** A special class that does one thing, the identity transform, in a
fast way.

## Class: _BlendedMixin

**Description:** Common methods for `BlendedGenericTransform` and `BlendedAffine2D`.

## Class: BlendedGenericTransform

**Description:** A "blended" transform uses one transform for the *x*-direction, and
another transform for the *y*-direction.

This "generic" version can handle any given child transform in the
*x*- and *y*-directions.

## Class: BlendedAffine2D

**Description:** A "blended" transform uses one transform for the *x*-direction, and
another transform for the *y*-direction.

This version is an optimization for the case where both child
transforms are of type `Affine2DBase`.

### Function: blended_transform_factory(x_transform, y_transform)

**Description:** Create a new "blended" transform using *x_transform* to transform
the *x*-axis and *y_transform* to transform the *y*-axis.

A faster version of the blended transform is returned for the case
where both child transforms are affine.

## Class: CompositeGenericTransform

**Description:** A composite transform formed by applying transform *a* then
transform *b*.

This "generic" version can handle any two arbitrary
transformations.

## Class: CompositeAffine2D

**Description:** A composite transform formed by applying transform *a* then transform *b*.

This version is an optimization that handles the case where both *a*
and *b* are 2D affines.

### Function: composite_transform_factory(a, b)

**Description:** Create a new composite transform that is the result of applying
transform a then transform b.

Shortcut versions of the blended transform are provided for the
case where both child transforms are affine, or one or the other
is the identity transform.

Composite transforms may also be created using the '+' operator,
e.g.::

  c = a + b

## Class: BboxTransform

**Description:** `BboxTransform` linearly transforms points from one `Bbox` to another.

## Class: BboxTransformTo

**Description:** `BboxTransformTo` is a transformation that linearly transforms points from
the unit bounding box to a given `Bbox`.

## Class: BboxTransformToMaxOnly

**Description:** `BboxTransformToMaxOnly` is a transformation that linearly transforms points from
the unit bounding box to a given `Bbox` with a fixed upper left of (0, 0).

## Class: BboxTransformFrom

**Description:** `BboxTransformFrom` linearly transforms points from a given `Bbox` to the
unit bounding box.

## Class: ScaledTranslation

**Description:** A transformation that translates by *xt* and *yt*, after *xt* and *yt*
have been transformed by *scale_trans*.

## Class: _ScaledRotation

**Description:** A transformation that applies rotation by *theta*, after transform by *trans_shift*.

## Class: AffineDeltaTransform

**Description:** A transform wrapper for transforming displacements between pairs of points.

This class is intended to be used to transform displacements ("position
deltas") between pairs of points (e.g., as the ``offset_transform``
of `.Collection`\s): given a transform ``t`` such that ``t =
AffineDeltaTransform(t) + offset``, ``AffineDeltaTransform``
satisfies ``AffineDeltaTransform(a - b) == AffineDeltaTransform(a) -
AffineDeltaTransform(b)``.

This is implemented by forcing the offset components of the transform
matrix to zero.

This class is experimental as of 3.3, and the API may change.

## Class: TransformedPath

**Description:** A `TransformedPath` caches a non-affine transformed copy of the
`~.path.Path`.  This cached copy is automatically updated when the
non-affine part of the transform changes.

.. note::

    Paths are considered immutable by this class. Any update to the
    path's vertices/codes will not trigger a transform recomputation.

## Class: TransformedPatchPath

**Description:** A `TransformedPatchPath` caches a non-affine transformed copy of the
`~.patches.Patch`. This cached copy is automatically updated when the
non-affine part of the transform or the patch changes.

### Function: nonsingular(vmin, vmax, expander, tiny, increasing)

**Description:** Modify the endpoints of a range as needed to avoid singularities.

Parameters
----------
vmin, vmax : float
    The initial endpoints.
expander : float, default: 0.001
    Fractional amount by which *vmin* and *vmax* are expanded if
    the original interval is too small, based on *tiny*.
tiny : float, default: 1e-15
    Threshold for the ratio of the interval to the maximum absolute
    value of its endpoints.  If the interval is smaller than
    this, it will be expanded.  This value should be around
    1e-15 or larger; otherwise the interval will be approaching
    the double precision resolution limit.
increasing : bool, default: True
    If True, swap *vmin*, *vmax* if *vmin* > *vmax*.

Returns
-------
vmin, vmax : float
    Endpoints, expanded and/or swapped if necessary.
    If either input is inf or NaN, or if both inputs are 0 or very
    close to zero, it returns -*expander*, *expander*.

### Function: interval_contains(interval, val)

**Description:** Check, inclusively, whether an interval includes a given value.

Parameters
----------
interval : (float, float)
    The endpoints of the interval.
val : float
    Value to check is within interval.

Returns
-------
bool
    Whether *val* is within the *interval*.

### Function: _interval_contains_close(interval, val, rtol)

**Description:** Check, inclusively, whether an interval includes a given value, with the
interval expanded by a small tolerance to admit floating point errors.

Parameters
----------
interval : (float, float)
    The endpoints of the interval.
val : float
    Value to check is within interval.
rtol : float, default: 1e-10
    Relative tolerance slippage allowed outside of the interval.
    For an interval ``[a, b]``, values
    ``a - rtol * (b - a) <= val <= b + rtol * (b - a)`` are considered
    inside the interval.

Returns
-------
bool
    Whether *val* is within the *interval* (with tolerance).

### Function: interval_contains_open(interval, val)

**Description:** Check, excluding endpoints, whether an interval includes a given value.

Parameters
----------
interval : (float, float)
    The endpoints of the interval.
val : float
    Value to check is within interval.

Returns
-------
bool
    Whether *val* is within the *interval*.

### Function: offset_copy(trans, fig, x, y, units)

**Description:** Return a new transform with an added offset.

Parameters
----------
trans : `Transform` subclass
    Any transform, to which offset will be applied.
fig : `~matplotlib.figure.Figure`, default: None
    Current figure. It can be None if *units* are 'dots'.
x, y : float, default: 0.0
    The offset to apply.
units : {'inches', 'points', 'dots'}, default: 'inches'
    Units of the offset.

Returns
-------
`Transform` subclass
    Transform with applied offset.

### Function: strrepr(x)

### Function: __init__(self, shorthand_name)

**Description:** Parameters
----------
shorthand_name : str
    A string representing the "name" of the transform. The name carries
    no significance other than to improve the readability of
    ``str(transform)`` when DEBUG=True.

### Function: __getstate__(self)

### Function: __setstate__(self, data_dict)

### Function: __copy__(self)

### Function: invalidate(self)

**Description:** Invalidate this `TransformNode` and triggers an invalidation of its
ancestors.  Should be called any time the transform changes.

### Function: _invalidate_internal(self, level, invalidating_node)

**Description:** Called by :meth:`invalidate` and subsequently ascends the transform
stack calling each TransformNode's _invalidate_internal method.

### Function: set_children(self)

**Description:** Set the children of the transform, to let the invalidation
system know which transforms can invalidate this transform.
Should be called from the constructor of any transforms that
depend on other transforms.

### Function: frozen(self)

**Description:** Return a frozen copy of this transform node.  The frozen copy will not
be updated when its children change.  Useful for storing a previously
known state of a transform where ``copy.deepcopy()`` might normally be
used.

### Function: frozen(self)

### Function: __array__(self)

### Function: x0(self)

**Description:** The first of the pair of *x* coordinates that define the bounding box.

This is not guaranteed to be less than :attr:`x1` (for that, use
:attr:`xmin`).

### Function: y0(self)

**Description:** The first of the pair of *y* coordinates that define the bounding box.

This is not guaranteed to be less than :attr:`y1` (for that, use
:attr:`ymin`).

### Function: x1(self)

**Description:** The second of the pair of *x* coordinates that define the bounding box.

This is not guaranteed to be greater than :attr:`x0` (for that, use
:attr:`xmax`).

### Function: y1(self)

**Description:** The second of the pair of *y* coordinates that define the bounding box.

This is not guaranteed to be greater than :attr:`y0` (for that, use
:attr:`ymax`).

### Function: p0(self)

**Description:** The first pair of (*x*, *y*) coordinates that define the bounding box.

This is not guaranteed to be the bottom-left corner (for that, use
:attr:`min`).

### Function: p1(self)

**Description:** The second pair of (*x*, *y*) coordinates that define the bounding box.

This is not guaranteed to be the top-right corner (for that, use
:attr:`max`).

### Function: xmin(self)

**Description:** The left edge of the bounding box.

### Function: ymin(self)

**Description:** The bottom edge of the bounding box.

### Function: xmax(self)

**Description:** The right edge of the bounding box.

### Function: ymax(self)

**Description:** The top edge of the bounding box.

### Function: min(self)

**Description:** The bottom-left corner of the bounding box.

### Function: max(self)

**Description:** The top-right corner of the bounding box.

### Function: intervalx(self)

**Description:** The pair of *x* coordinates that define the bounding box.

This is not guaranteed to be sorted from left to right.

### Function: intervaly(self)

**Description:** The pair of *y* coordinates that define the bounding box.

This is not guaranteed to be sorted from bottom to top.

### Function: width(self)

**Description:** The (signed) width of the bounding box.

### Function: height(self)

**Description:** The (signed) height of the bounding box.

### Function: size(self)

**Description:** The (signed) width and height of the bounding box.

### Function: bounds(self)

**Description:** Return (:attr:`x0`, :attr:`y0`, :attr:`width`, :attr:`height`).

### Function: extents(self)

**Description:** Return (:attr:`x0`, :attr:`y0`, :attr:`x1`, :attr:`y1`).

### Function: get_points(self)

### Function: containsx(self, x)

**Description:** Return whether *x* is in the closed (:attr:`x0`, :attr:`x1`) interval.

### Function: containsy(self, y)

**Description:** Return whether *y* is in the closed (:attr:`y0`, :attr:`y1`) interval.

### Function: contains(self, x, y)

**Description:** Return whether ``(x, y)`` is in the bounding box or on its edge.

### Function: overlaps(self, other)

**Description:** Return whether this bounding box overlaps with the other bounding box.

Parameters
----------
other : `.BboxBase`

### Function: fully_containsx(self, x)

**Description:** Return whether *x* is in the open (:attr:`x0`, :attr:`x1`) interval.

### Function: fully_containsy(self, y)

**Description:** Return whether *y* is in the open (:attr:`y0`, :attr:`y1`) interval.

### Function: fully_contains(self, x, y)

**Description:** Return whether ``x, y`` is in the bounding box, but not on its edge.

### Function: fully_overlaps(self, other)

**Description:** Return whether this bounding box overlaps with the other bounding box,
not including the edges.

Parameters
----------
other : `.BboxBase`

### Function: transformed(self, transform)

**Description:** Construct a `Bbox` by statically transforming this one by *transform*.

### Function: anchored(self, c, container)

**Description:** Return a copy of the `Bbox` anchored to *c* within *container*.

Parameters
----------
c : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
    Either an (*x*, *y*) pair of relative coordinates (0 is left or
    bottom, 1 is right or top), 'C' (center), or a cardinal direction
    ('SW', southwest, is bottom left, etc.).
container : `Bbox`
    The box within which the `Bbox` is positioned.

See Also
--------
.Axes.set_anchor

### Function: shrunk(self, mx, my)

**Description:** Return a copy of the `Bbox`, shrunk by the factor *mx*
in the *x* direction and the factor *my* in the *y* direction.
The lower left corner of the box remains unchanged.  Normally
*mx* and *my* will be less than 1, but this is not enforced.

### Function: shrunk_to_aspect(self, box_aspect, container, fig_aspect)

**Description:** Return a copy of the `Bbox`, shrunk so that it is as
large as it can be while having the desired aspect ratio,
*box_aspect*.  If the box coordinates are relative (i.e.
fractions of a larger box such as a figure) then the
physical aspect ratio of that figure is specified with
*fig_aspect*, so that *box_aspect* can also be given as a
ratio of the absolute dimensions, not the relative dimensions.

### Function: splitx(self)

**Description:** Return a list of new `Bbox` objects formed by splitting the original
one with vertical lines at fractional positions given by *args*.

### Function: splity(self)

**Description:** Return a list of new `Bbox` objects formed by splitting the original
one with horizontal lines at fractional positions given by *args*.

### Function: count_contains(self, vertices)

**Description:** Count the number of vertices contained in the `Bbox`.
Any vertices with a non-finite x or y value are ignored.

Parameters
----------
vertices : (N, 2) array

### Function: count_overlaps(self, bboxes)

**Description:** Count the number of bounding boxes that overlap this one.

Parameters
----------
bboxes : sequence of `.BboxBase`

### Function: expanded(self, sw, sh)

**Description:** Construct a `Bbox` by expanding this one around its center by the
factors *sw* and *sh*.

### Function: padded(self, w_pad, h_pad)

**Description:** Construct a `Bbox` by padding this one on all four sides.

Parameters
----------
w_pad : float
    Width pad
h_pad : float, optional
    Height pad.  Defaults to *w_pad*.

### Function: translated(self, tx, ty)

**Description:** Construct a `Bbox` by translating this one by *tx* and *ty*.

### Function: corners(self)

**Description:** Return the corners of this rectangle as an array of points.

Specifically, this returns the array
``[[x0, y0], [x0, y1], [x1, y0], [x1, y1]]``.

### Function: rotated(self, radians)

**Description:** Return the axes-aligned bounding box that bounds the result of rotating
this `Bbox` by an angle of *radians*.

### Function: union(bboxes)

**Description:** Return a `Bbox` that contains all of the given *bboxes*.

### Function: intersection(bbox1, bbox2)

**Description:** Return the intersection of *bbox1* and *bbox2* if they intersect, or
None if they don't.

### Function: __init__(self, points)

**Description:** Parameters
----------
points : `~numpy.ndarray`
    A (2, 2) array of the form ``[[x0, y0], [x1, y1]]``.

### Function: frozen(self)

### Function: unit()

**Description:** Create a new unit `Bbox` from (0, 0) to (1, 1).

### Function: null()

**Description:** Create a new null `Bbox` from (inf, inf) to (-inf, -inf).

### Function: from_bounds(x0, y0, width, height)

**Description:** Create a new `Bbox` from *x0*, *y0*, *width* and *height*.

*width* and *height* may be negative.

### Function: from_extents()

**Description:** Create a new Bbox from *left*, *bottom*, *right* and *top*.

The *y*-axis increases upwards.

Parameters
----------
left, bottom, right, top : float
    The four extents of the bounding box.
minpos : float or None
    If this is supplied, the Bbox will have a minimum positive value
    set. This is useful when dealing with logarithmic scales and other
    scales where negative bounds result in floating point errors.

### Function: __format__(self, fmt)

### Function: __str__(self)

### Function: __repr__(self)

### Function: ignore(self, value)

**Description:** Set whether the existing bounds of the box should be ignored
by subsequent calls to :meth:`update_from_data_xy`.

value : bool
    - When ``True``, subsequent calls to `update_from_data_xy` will
      ignore the existing bounds of the `Bbox`.
    - When ``False``, subsequent calls to `update_from_data_xy` will
      include the existing bounds of the `Bbox`.

### Function: update_from_path(self, path, ignore, updatex, updatey)

**Description:** Update the bounds of the `Bbox` to contain the vertices of the
provided path. After updating, the bounds will have positive *width*
and *height*; *x0* and *y0* will be the minimal values.

Parameters
----------
path : `~matplotlib.path.Path`
ignore : bool, optional
   - When ``True``, ignore the existing bounds of the `Bbox`.
   - When ``False``, include the existing bounds of the `Bbox`.
   - When ``None``, use the last value passed to :meth:`ignore`.
updatex, updatey : bool, default: True
    When ``True``, update the x/y values.

### Function: update_from_data_x(self, x, ignore)

**Description:** Update the x-bounds of the `Bbox` based on the passed in data. After
updating, the bounds will have positive *width*, and *x0* will be the
minimal value.

Parameters
----------
x : `~numpy.ndarray`
    Array of x-values.
ignore : bool, optional
   - When ``True``, ignore the existing bounds of the `Bbox`.
   - When ``False``, include the existing bounds of the `Bbox`.
   - When ``None``, use the last value passed to :meth:`ignore`.

### Function: update_from_data_y(self, y, ignore)

**Description:** Update the y-bounds of the `Bbox` based on the passed in data. After
updating, the bounds will have positive *height*, and *y0* will be the
minimal value.

Parameters
----------
y : `~numpy.ndarray`
    Array of y-values.
ignore : bool, optional
    - When ``True``, ignore the existing bounds of the `Bbox`.
    - When ``False``, include the existing bounds of the `Bbox`.
    - When ``None``, use the last value passed to :meth:`ignore`.

### Function: update_from_data_xy(self, xy, ignore, updatex, updatey)

**Description:** Update the `Bbox` bounds based on the passed in *xy* coordinates.

After updating, the bounds will have positive *width* and *height*;
*x0* and *y0* will be the minimal values.

Parameters
----------
xy : (N, 2) array-like
    The (x, y) coordinates.
ignore : bool, optional
    - When ``True``, ignore the existing bounds of the `Bbox`.
    - When ``False``, include the existing bounds of the `Bbox`.
    - When ``None``, use the last value passed to :meth:`ignore`.
updatex, updatey : bool, default: True
     When ``True``, update the x/y values.

### Function: x0(self, val)

### Function: y0(self, val)

### Function: x1(self, val)

### Function: y1(self, val)

### Function: p0(self, val)

### Function: p1(self, val)

### Function: intervalx(self, interval)

### Function: intervaly(self, interval)

### Function: bounds(self, bounds)

### Function: minpos(self)

**Description:** The minimum positive value in both directions within the Bbox.

This is useful when dealing with logarithmic scales and other scales
where negative bounds result in floating point errors, and will be used
as the minimum extent instead of *p0*.

### Function: minpos(self, val)

### Function: minposx(self)

**Description:** The minimum positive value in the *x*-direction within the Bbox.

This is useful when dealing with logarithmic scales and other scales
where negative bounds result in floating point errors, and will be used
as the minimum *x*-extent instead of *x0*.

### Function: minposx(self, val)

### Function: minposy(self)

**Description:** The minimum positive value in the *y*-direction within the Bbox.

This is useful when dealing with logarithmic scales and other scales
where negative bounds result in floating point errors, and will be used
as the minimum *y*-extent instead of *y0*.

### Function: minposy(self, val)

### Function: get_points(self)

**Description:** Get the points of the bounding box as an array of the form
``[[x0, y0], [x1, y1]]``.

### Function: set_points(self, points)

**Description:** Set the points of the bounding box directly from an array of the form
``[[x0, y0], [x1, y1]]``.  No error checking is performed, as this
method is mainly for internal use.

### Function: set(self, other)

**Description:** Set this bounding box from the "frozen" bounds of another `Bbox`.

### Function: mutated(self)

**Description:** Return whether the bbox has changed since init.

### Function: mutatedx(self)

**Description:** Return whether the x-limits have changed since init.

### Function: mutatedy(self)

**Description:** Return whether the y-limits have changed since init.

### Function: __init__(self, bbox, transform)

**Description:** Parameters
----------
bbox : `Bbox`
transform : `Transform`

### Function: get_points(self)

### Function: contains(self, x, y)

### Function: fully_contains(self, x, y)

### Function: __init__(self, bbox, x0, y0, x1, y1)

**Description:** Parameters
----------
bbox : `Bbox`
    The child bounding box to wrap.

x0 : float or None
    The locked value for x0, or None to leave unlocked.

y0 : float or None
    The locked value for y0, or None to leave unlocked.

x1 : float or None
    The locked value for x1, or None to leave unlocked.

y1 : float or None
    The locked value for y1, or None to leave unlocked.

### Function: get_points(self)

### Function: locked_x0(self)

**Description:** float or None: The value used for the locked x0.

### Function: locked_x0(self, x0)

### Function: locked_y0(self)

**Description:** float or None: The value used for the locked y0.

### Function: locked_y0(self, y0)

### Function: locked_x1(self)

**Description:** float or None: The value used for the locked x1.

### Function: locked_x1(self, x1)

### Function: locked_y1(self)

**Description:** float or None: The value used for the locked y1.

### Function: locked_y1(self, y1)

### Function: __init_subclass__(cls)

### Function: __add__(self, other)

**Description:** Compose two transforms together so that *self* is followed by *other*.

``A + B`` returns a transform ``C`` so that
``C.transform(x) == B.transform(A.transform(x))``.

### Function: _iter_break_from_left_to_right(self)

**Description:** Return an iterator breaking down this transform stack from left to
right recursively. If self == ((A, N), A) then the result will be an
iterator which yields I : ((A, N), A), followed by A : (N, A),
followed by (A, N) : (A), but not ((A, N), A) : I.

This is equivalent to flattening the stack then yielding
``flat_stack[:i], flat_stack[i:]`` where i=0..(n-1).

### Function: depth(self)

**Description:** Return the number of transforms which have been chained
together to form this Transform instance.

.. note::

    For the special case of a Composite transform, the maximum depth
    of the two is returned.

### Function: contains_branch(self, other)

**Description:** Return whether the given transform is a sub-tree of this transform.

This routine uses transform equality to identify sub-trees, therefore
in many situations it is object id which will be used.

For the case where the given transform represents the whole
of this transform, returns True.

### Function: contains_branch_seperately(self, other_transform)

**Description:** Return whether the given branch is a sub-tree of this transform on
each separate dimension.

A common use for this method is to identify if a transform is a blended
transform containing an Axes' data transform. e.g.::

    x_isdata, y_isdata = trans.contains_branch_seperately(ax.transData)

### Function: __sub__(self, other)

**Description:** Compose *self* with the inverse of *other*, cancelling identical terms
if any::

    # In general:
    A - B == A + B.inverted()
    # (but see note regarding frozen transforms below).

    # If A "ends with" B (i.e. A == A' + B for some A') we can cancel
    # out B:
    (A' + B) - B == A'

    # Likewise, if B "starts with" A (B = A + B'), we can cancel out A:
    A - (A + B') == B'.inverted() == B'^-1

Cancellation (rather than naively returning ``A + B.inverted()``) is
important for multiple reasons:

- It avoids floating-point inaccuracies when computing the inverse of
  B: ``B - B`` is guaranteed to cancel out exactly (resulting in the
  identity transform), whereas ``B + B.inverted()`` may differ by a
  small epsilon.
- ``B.inverted()`` always returns a frozen transform: if one computes
  ``A + B + B.inverted()`` and later mutates ``B``, then
  ``B.inverted()`` won't be updated and the last two terms won't cancel
  out anymore; on the other hand, ``A + B - B`` will always be equal to
  ``A`` even if ``B`` is mutated.

### Function: __array__(self)

**Description:** Array interface to get at this Transform's affine matrix.

### Function: transform(self, values)

**Description:** Apply this transformation on the given array of *values*.

Parameters
----------
values : array-like
    The input values as an array of length :attr:`input_dims` or
    shape (N, :attr:`input_dims`).

Returns
-------
array
    The output values as an array of length :attr:`output_dims` or
    shape (N, :attr:`output_dims`), depending on the input.

### Function: transform_affine(self, values)

**Description:** Apply only the affine part of this transformation on the
given array of values.

``transform(values)`` is always equivalent to
``transform_affine(transform_non_affine(values))``.

In non-affine transformations, this is generally a no-op.  In
affine transformations, this is equivalent to
``transform(values)``.

Parameters
----------
values : array
    The input values as an array of length :attr:`input_dims` or
    shape (N, :attr:`input_dims`).

Returns
-------
array
    The output values as an array of length :attr:`output_dims` or
    shape (N, :attr:`output_dims`), depending on the input.

### Function: transform_non_affine(self, values)

**Description:** Apply only the non-affine part of this transformation.

``transform(values)`` is always equivalent to
``transform_affine(transform_non_affine(values))``.

In non-affine transformations, this is generally equivalent to
``transform(values)``.  In affine transformations, this is
always a no-op.

Parameters
----------
values : array
    The input values as an array of length :attr:`input_dims` or
    shape (N, :attr:`input_dims`).

Returns
-------
array
    The output values as an array of length :attr:`output_dims` or
    shape (N, :attr:`output_dims`), depending on the input.

### Function: transform_bbox(self, bbox)

**Description:** Transform the given bounding box.

For smarter transforms including caching (a common requirement in
Matplotlib), see `TransformedBbox`.

### Function: get_affine(self)

**Description:** Get the affine part of this transform.

### Function: get_matrix(self)

**Description:** Get the matrix for the affine part of this transform.

### Function: transform_point(self, point)

**Description:** Return a transformed point.

This function is only kept for backcompatibility; the more general
`.transform` method is capable of transforming both a list of points
and a single point.

The point is given as a sequence of length :attr:`input_dims`.
The transformed point is returned as a sequence of length
:attr:`output_dims`.

### Function: transform_path(self, path)

**Description:** Apply the transform to `.Path` *path*, returning a new `.Path`.

In some cases, this transform may insert curves into the path
that began as line segments.

### Function: transform_path_affine(self, path)

**Description:** Apply the affine part of this transform to `.Path` *path*, returning a
new `.Path`.

``transform_path(path)`` is equivalent to
``transform_path_affine(transform_path_non_affine(values))``.

### Function: transform_path_non_affine(self, path)

**Description:** Apply the non-affine part of this transform to `.Path` *path*,
returning a new `.Path`.

``transform_path(path)`` is equivalent to
``transform_path_affine(transform_path_non_affine(values))``.

### Function: transform_angles(self, angles, pts, radians, pushoff)

**Description:** Transform a set of angles anchored at specific locations.

Parameters
----------
angles : (N,) array-like
    The angles to transform.
pts : (N, 2) array-like
    The points where the angles are anchored.
radians : bool, default: False
    Whether *angles* are radians or degrees.
pushoff : float
    For each point in *pts* and angle in *angles*, the transformed
    angle is computed by transforming a segment of length *pushoff*
    starting at that point and making that angle relative to the
    horizontal axis, and measuring the angle between the horizontal
    axis and the transformed segment.

Returns
-------
(N,) array

### Function: inverted(self)

**Description:** Return the corresponding inverse transformation.

It holds ``x == self.inverted().transform(self.transform(x))``.

The return value of this method should be treated as
temporary.  An update to *self* does not cause a corresponding
update to its inverted copy.

### Function: __init__(self, child)

**Description:** *child*: A `Transform` instance.  This child may later
be replaced with :meth:`set`.

### Function: __eq__(self, other)

### Function: frozen(self)

### Function: set(self, child)

**Description:** Replace the current child of this transform with another one.

The new child must have the same number of input and output
dimensions as the current child.

### Function: __init__(self)

### Function: __array__(self)

### Function: __eq__(self, other)

### Function: transform(self, values)

### Function: transform_affine(self, values)

### Function: transform_non_affine(self, values)

### Function: transform_path(self, path)

### Function: transform_path_affine(self, path)

### Function: transform_path_non_affine(self, path)

### Function: get_affine(self)

### Function: frozen(self)

### Function: is_separable(self)

### Function: to_values(self)

**Description:** Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.

### Function: transform_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, matrix)

**Description:** Initialize an Affine transform from a 3x3 numpy float array::

  a c e
  b d f
  0 0 1

If *matrix* is None, initialize with the identity transform.

### Function: __str__(self)

### Function: from_values(a, b, c, d, e, f)

**Description:** Create a new Affine2D instance from the given values::

  a c e
  b d f
  0 0 1

.

### Function: get_matrix(self)

**Description:** Get the underlying transformation matrix as a 3x3 array::

  a c e
  b d f
  0 0 1

.

### Function: set_matrix(self, mtx)

**Description:** Set the underlying transformation matrix from a 3x3 array::

  a c e
  b d f
  0 0 1

.

### Function: set(self, other)

**Description:** Set this transformation from the frozen copy of another
`Affine2DBase` object.

### Function: clear(self)

**Description:** Reset the underlying matrix to the identity transform.

### Function: rotate(self, theta)

**Description:** Add a rotation (in radians) to this transform in place.

Returns *self*, so this method can easily be chained with more
calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
and :meth:`scale`.

### Function: rotate_deg(self, degrees)

**Description:** Add a rotation (in degrees) to this transform in place.

Returns *self*, so this method can easily be chained with more
calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
and :meth:`scale`.

### Function: rotate_around(self, x, y, theta)

**Description:** Add a rotation (in radians) around the point (x, y) in place.

Returns *self*, so this method can easily be chained with more
calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
and :meth:`scale`.

### Function: rotate_deg_around(self, x, y, degrees)

**Description:** Add a rotation (in degrees) around the point (x, y) in place.

Returns *self*, so this method can easily be chained with more
calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
and :meth:`scale`.

### Function: translate(self, tx, ty)

**Description:** Add a translation in place.

Returns *self*, so this method can easily be chained with more
calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
and :meth:`scale`.

### Function: scale(self, sx, sy)

**Description:** Add a scale in place.

If *sy* is None, the same scale is applied in both the *x*- and
*y*-directions.

Returns *self*, so this method can easily be chained with more
calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
and :meth:`scale`.

### Function: skew(self, xShear, yShear)

**Description:** Add a skew in place.

*xShear* and *yShear* are the shear angles along the *x*- and
*y*-axes, respectively, in radians.

Returns *self*, so this method can easily be chained with more
calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
and :meth:`scale`.

### Function: skew_deg(self, xShear, yShear)

**Description:** Add a skew in place.

*xShear* and *yShear* are the shear angles along the *x*- and
*y*-axes, respectively, in degrees.

Returns *self*, so this method can easily be chained with more
calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
and :meth:`scale`.

### Function: frozen(self)

### Function: get_matrix(self)

### Function: transform(self, values)

### Function: transform_affine(self, values)

### Function: transform_non_affine(self, values)

### Function: transform_path(self, path)

### Function: transform_path_affine(self, path)

### Function: transform_path_non_affine(self, path)

### Function: get_affine(self)

### Function: inverted(self)

### Function: __eq__(self, other)

### Function: contains_branch_seperately(self, transform)

### Function: __init__(self, x_transform, y_transform)

**Description:** Create a new "blended" transform using *x_transform* to transform the
*x*-axis and *y_transform* to transform the *y*-axis.

You will generally not call this constructor directly but use the
`blended_transform_factory` function instead, which can determine
automatically which kind of blended transform to create.

### Function: depth(self)

### Function: contains_branch(self, other)

### Function: frozen(self)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: get_affine(self)

### Function: __init__(self, x_transform, y_transform)

**Description:** Create a new "blended" transform using *x_transform* to transform the
*x*-axis and *y_transform* to transform the *y*-axis.

Both *x_transform* and *y_transform* must be 2D affine transforms.

You will generally not call this constructor directly but use the
`blended_transform_factory` function instead, which can determine
automatically which kind of blended transform to create.

### Function: get_matrix(self)

### Function: __init__(self, a, b)

**Description:** Create a new composite transform that is the result of
applying transform *a* then transform *b*.

You will generally not call this constructor directly but write ``a +
b`` instead, which will automatically choose the best kind of composite
transform instance to create.

### Function: frozen(self)

### Function: _invalidate_internal(self, level, invalidating_node)

### Function: __eq__(self, other)

### Function: _iter_break_from_left_to_right(self)

### Function: contains_branch_seperately(self, other_transform)

### Function: transform_affine(self, values)

### Function: transform_non_affine(self, values)

### Function: transform_path_non_affine(self, path)

### Function: get_affine(self)

### Function: inverted(self)

### Function: __init__(self, a, b)

**Description:** Create a new composite transform that is the result of
applying `Affine2DBase` *a* then `Affine2DBase` *b*.

You will generally not call this constructor directly but write ``a +
b`` instead, which will automatically choose the best kind of composite
transform instance to create.

### Function: depth(self)

### Function: _iter_break_from_left_to_right(self)

### Function: get_matrix(self)

### Function: __init__(self, boxin, boxout)

**Description:** Create a new `BboxTransform` that linearly transforms
points from *boxin* to *boxout*.

### Function: get_matrix(self)

### Function: __init__(self, boxout)

**Description:** Create a new `BboxTransformTo` that linearly transforms
points from the unit bounding box to *boxout*.

### Function: get_matrix(self)

### Function: get_matrix(self)

### Function: __init__(self, boxin)

### Function: get_matrix(self)

### Function: __init__(self, xt, yt, scale_trans)

### Function: get_matrix(self)

### Function: __init__(self, theta, trans_shift)

### Function: get_matrix(self)

### Function: __init__(self, transform)

### Function: get_matrix(self)

### Function: __init__(self, path, transform)

**Description:** Parameters
----------
path : `~.path.Path`
transform : `Transform`

### Function: _revalidate(self)

### Function: get_transformed_points_and_affine(self)

**Description:** Return a copy of the child path, with the non-affine part of
the transform already applied, along with the affine part of
the path necessary to complete the transformation.  Unlike
:meth:`get_transformed_path_and_affine`, no interpolation will
be performed.

### Function: get_transformed_path_and_affine(self)

**Description:** Return a copy of the child path, with the non-affine part of
the transform already applied, along with the affine part of
the path necessary to complete the transformation.

### Function: get_fully_transformed_path(self)

**Description:** Return a fully-transformed copy of the child path.

### Function: get_affine(self)

### Function: __init__(self, patch)

**Description:** Parameters
----------
patch : `~.patches.Patch`

### Function: _revalidate(self)

### Function: __str__(self)

### Function: _check(points)

### Function: __init__(self, points)

### Function: invalidate(self)

### Function: get_points(self)

### Function: get_points(self)

### Function: transform_affine(self, values)
