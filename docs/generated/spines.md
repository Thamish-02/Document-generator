## AI Summary

A file named spines.py.


## Class: Spine

**Description:** An axis spine -- the line noting the data area boundaries.

Spines are the lines connecting the axis tick marks and noting the
boundaries of the data area. They can be placed at arbitrary
positions. See `~.Spine.set_position` for more information.

The default position is ``('outward', 0)``.

Spines are subclasses of `.Patch`, and inherit much of their behavior.

Spines draw a line, a circle, or an arc depending on if
`~.Spine.set_patch_line`, `~.Spine.set_patch_circle`, or
`~.Spine.set_patch_arc` has been called. Line-like is the default.

For examples see :ref:`spines_examples`.

## Class: SpinesProxy

**Description:** A proxy to broadcast ``set_*()`` and ``set()`` method calls to contained `.Spines`.

The proxy cannot be used for any other operations on its members.

The supported methods are determined dynamically based on the contained
spines. If not all spines support a given method, it's executed only on
the subset of spines that support it.

## Class: Spines

**Description:** The container of all `.Spine`\s in an Axes.

The interface is dict-like mapping names (e.g. 'left') to `.Spine` objects.
Additionally, it implements some pandas.Series-like features like accessing
elements by attribute::

    spines['top'].set_visible(False)
    spines.top.set_visible(False)

Multiple spines can be addressed simultaneously by passing a list::

    spines[['top', 'right']].set_visible(False)

Use an open slice to address all spines::

    spines[:].set_visible(False)

The latter two indexing methods will return a `SpinesProxy` that broadcasts all
``set_*()`` and ``set()`` calls to its members, but cannot be used for any other
operation.

### Function: __str__(self)

### Function: __init__(self, axes, spine_type, path)

**Description:** Parameters
----------
axes : `~matplotlib.axes.Axes`
    The `~.axes.Axes` instance containing the spine.
spine_type : str
    The spine type.
path : `~matplotlib.path.Path`
    The `.Path` instance used to draw the spine.

Other Parameters
----------------
**kwargs
    Valid keyword arguments are:

    %(Patch:kwdoc)s

### Function: set_patch_arc(self, center, radius, theta1, theta2)

**Description:** Set the spine to be arc-like.

### Function: set_patch_circle(self, center, radius)

**Description:** Set the spine to be circular.

### Function: set_patch_line(self)

**Description:** Set the spine to be linear.

### Function: _recompute_transform(self)

**Description:** Notes
-----
This cannot be called until after this has been added to an Axes,
otherwise unit conversion will fail. This makes it very important to
call the accessor method and not directly access the transformation
member variable.

### Function: get_patch_transform(self)

### Function: get_window_extent(self, renderer)

**Description:** Return the window extent of the spines in display space, including
padding for ticks (but not their labels)

See Also
--------
matplotlib.axes.Axes.get_tightbbox
matplotlib.axes.Axes.get_window_extent

### Function: get_path(self)

### Function: _ensure_position_is_set(self)

### Function: register_axis(self, axis)

**Description:** Register an axis.

An axis should be registered with its corresponding spine from
the Axes instance. This allows the spine to clear any axis
properties when needed.

### Function: clear(self)

**Description:** Clear the current spine.

### Function: _clear(self)

**Description:** Clear things directly related to the spine.

In this way it is possible to avoid clearing the Axis as well when calling
from library code where it is known that the Axis is cleared separately.

### Function: _adjust_location(self)

**Description:** Automatically set spine bounds to the view interval.

### Function: draw(self, renderer)

### Function: set_position(self, position)

**Description:** Set the position of the spine.

Spine position is specified by a 2 tuple of (position type,
amount). The position types are:

* 'outward': place the spine out from the data area by the specified
  number of points. (Negative values place the spine inwards.)
* 'axes': place the spine at the specified Axes coordinate (0 to 1).
* 'data': place the spine at the specified data coordinate.

Additionally, shorthand notations define a special positions:

* 'center' -> ``('axes', 0.5)``
* 'zero' -> ``('data', 0.0)``

Examples
--------
:doc:`/gallery/spines/spine_placement_demo`

### Function: get_position(self)

**Description:** Return the spine position.

### Function: get_spine_transform(self)

**Description:** Return the spine transform.

### Function: set_bounds(self, low, high)

**Description:** Set the spine bounds.

Parameters
----------
low : float or None, optional
    The lower spine bound. Passing *None* leaves the limit unchanged.

    The bounds may also be passed as the tuple (*low*, *high*) as the
    first positional argument.

    .. ACCEPTS: (low: float, high: float)

high : float or None, optional
    The higher spine bound. Passing *None* leaves the limit unchanged.

### Function: get_bounds(self)

**Description:** Get the bounds of the spine.

### Function: linear_spine(cls, axes, spine_type)

**Description:** Create and return a linear `Spine`.

### Function: arc_spine(cls, axes, spine_type, center, radius, theta1, theta2)

**Description:** Create and return an arc `Spine`.

### Function: circular_spine(cls, axes, center, radius)

**Description:** Create and return a circular `Spine`.

### Function: set_color(self, c)

**Description:** Set the edgecolor.

Parameters
----------
c : :mpltype:`color`

Notes
-----
This method does not modify the facecolor (which defaults to "none"),
unlike the `.Patch.set_color` method defined in the parent class.  Use
`.Patch.set_facecolor` to set the facecolor.

### Function: __init__(self, spine_dict)

### Function: __getattr__(self, name)

### Function: __dir__(self)

### Function: __init__(self)

### Function: from_dict(cls, d)

### Function: __getstate__(self)

### Function: __setstate__(self, state)

### Function: __getattr__(self, name)

### Function: __getitem__(self, key)

### Function: __setitem__(self, key, value)

### Function: __delitem__(self, key)

### Function: __iter__(self)

### Function: __len__(self)

### Function: x(_targets, _funcname)
