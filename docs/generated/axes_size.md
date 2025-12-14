## AI Summary

A file named axes_size.py.


## Class: _Base

## Class: Add

**Description:** Sum of two sizes.

## Class: Fixed

**Description:** Simple fixed size with absolute part = *fixed_size* and relative part = 0.

## Class: Scaled

**Description:** Simple scaled(?) size with absolute part = 0 and
relative part = *scalable_size*.

### Function: _get_axes_aspect(ax)

## Class: AxesX

**Description:** Scaled size whose relative part corresponds to the data width
of the *axes* multiplied by the *aspect*.

## Class: AxesY

**Description:** Scaled size whose relative part corresponds to the data height
of the *axes* multiplied by the *aspect*.

## Class: MaxExtent

**Description:** Size whose absolute part is either the largest width or the largest height
of the given *artist_list*.

## Class: MaxWidth

**Description:** Size whose absolute part is the largest width of the given *artist_list*.

## Class: MaxHeight

**Description:** Size whose absolute part is the largest height of the given *artist_list*.

## Class: Fraction

**Description:** An instance whose size is a *fraction* of the *ref_size*.

>>> s = Fraction(0.3, AxesX(ax))

### Function: from_any(size, fraction_ref)

**Description:** Create a Fixed unit when the first argument is a float, or a
Fraction unit if that is a string that ends with %. The second
argument is only meaningful when Fraction unit is created.

>>> from mpl_toolkits.axes_grid1.axes_size import from_any
>>> a = from_any(1.2) # => Fixed(1.2)
>>> from_any("50%", a) # => Fraction(0.5, a)

## Class: _AxesDecorationsSize

**Description:** Fixed size, corresponding to the size of decorations on a given Axes side.

### Function: __rmul__(self, other)

### Function: __mul__(self, other)

### Function: __div__(self, other)

### Function: __add__(self, other)

### Function: __neg__(self)

### Function: __radd__(self, other)

### Function: __sub__(self, other)

### Function: get_size(self, renderer)

**Description:** Return two-float tuple with relative and absolute sizes.

### Function: __init__(self, a, b)

### Function: get_size(self, renderer)

### Function: __init__(self, fixed_size)

### Function: get_size(self, renderer)

### Function: __init__(self, scalable_size)

### Function: get_size(self, renderer)

### Function: __init__(self, axes, aspect, ref_ax)

### Function: get_size(self, renderer)

### Function: __init__(self, axes, aspect, ref_ax)

### Function: get_size(self, renderer)

### Function: __init__(self, artist_list, w_or_h)

### Function: add_artist(self, a)

### Function: get_size(self, renderer)

### Function: __init__(self, artist_list)

### Function: __init__(self, artist_list)

### Function: __init__(self, fraction, ref_size)

### Function: get_size(self, renderer)

### Function: __init__(self, ax, direction)

### Function: get_size(self, renderer)
