## AI Summary

A file named _trifinder.py.


## Class: TriFinder

**Description:** Abstract base class for classes used to find the triangles of a
Triangulation in which (x, y) points lie.

Rather than instantiate an object of a class derived from TriFinder, it is
usually better to use the function `.Triangulation.get_trifinder`.

Derived classes implement __call__(x, y) where x and y are array-like point
coordinates of the same shape.

## Class: TrapezoidMapTriFinder

**Description:** `~matplotlib.tri.TriFinder` class implemented using the trapezoid
map algorithm from the book "Computational Geometry, Algorithms and
Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars
and O. Schwarzkopf.

The triangulation must be valid, i.e. it must not have duplicate points,
triangles formed from colinear points, or overlapping triangles.  The
algorithm has some tolerance to triangles formed from colinear points, but
this should not be relied upon.

### Function: __init__(self, triangulation)

### Function: __call__(self, x, y)

### Function: __init__(self, triangulation)

### Function: __call__(self, x, y)

**Description:** Return an array containing the indices of the triangles in which the
specified *x*, *y* points lie, or -1 for points that do not lie within
a triangle.

*x*, *y* are array-like x and y coordinates of the same shape and any
number of dimensions.

Returns integer array with the same shape and *x* and *y*.

### Function: _get_tree_stats(self)

**Description:** Return a python list containing the statistics about the node tree:
    0: number of nodes (tree size)
    1: number of unique nodes
    2: number of trapezoids (tree leaf nodes)
    3: number of unique trapezoids
    4: maximum parent count (max number of times a node is repeated in
           tree)
    5: maximum depth of tree (one more than the maximum number of
           comparisons needed to search through the tree)
    6: mean of all trapezoid depths (one more than the average number
           of comparisons needed to search through the tree)

### Function: _initialize(self)

**Description:** Initialize the underlying C++ object.  Can be called multiple times if,
for example, the triangulation is modified.

### Function: _print_tree(self)

**Description:** Print a text representation of the node tree, which is useful for
debugging purposes.
