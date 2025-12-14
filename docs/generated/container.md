## AI Summary

A file named container.py.


## Class: Container

**Description:** Base class for containers.

Containers are classes that collect semantically related Artists such as
the bars of a bar plot.

## Class: BarContainer

**Description:** Container for the artists of bar plots (e.g. created by `.Axes.bar`).

The container can be treated as a tuple of the *patches* themselves.
Additionally, you can access these and further parameters by the
attributes.

Attributes
----------
patches : list of :class:`~matplotlib.patches.Rectangle`
    The artists of the bars.

errorbar : None or :class:`~matplotlib.container.ErrorbarContainer`
    A container for the error bar artists if error bars are present.
    *None* otherwise.

datavalues : None or array-like
    The underlying data values corresponding to the bars.

orientation : {'vertical', 'horizontal'}, default: None
    If 'vertical', the bars are assumed to be vertical.
    If 'horizontal', the bars are assumed to be horizontal.

## Class: ErrorbarContainer

**Description:** Container for the artists of error bars (e.g. created by `.Axes.errorbar`).

The container can be treated as the *lines* tuple itself.
Additionally, you can access these and further parameters by the
attributes.

Attributes
----------
lines : tuple
    Tuple of ``(data_line, caplines, barlinecols)``.

    - data_line : A `~matplotlib.lines.Line2D` instance of x, y plot markers
      and/or line.
    - caplines : A tuple of `~matplotlib.lines.Line2D` instances of the error
      bar caps.
    - barlinecols : A tuple of `~matplotlib.collections.LineCollection` with the
      horizontal and vertical error ranges.

has_xerr, has_yerr : bool
    ``True`` if the errorbar has x/y errors.

## Class: StemContainer

**Description:** Container for the artists created in a :meth:`.Axes.stem` plot.

The container can be treated like a namedtuple ``(markerline, stemlines,
baseline)``.

Attributes
----------
markerline : `~matplotlib.lines.Line2D`
    The artist of the markers at the stem heads.

stemlines : `~matplotlib.collections.LineCollection`
    The artists of the vertical lines for all stems.

baseline : `~matplotlib.lines.Line2D`
    The artist of the horizontal baseline.

### Function: __repr__(self)

### Function: __new__(cls)

### Function: __init__(self, kl, label)

### Function: remove(self)

### Function: get_children(self)

### Function: __init__(self, patches, errorbar)

### Function: __init__(self, lines, has_xerr, has_yerr)

### Function: __init__(self, markerline_stemlines_baseline)

**Description:** Parameters
----------
markerline_stemlines_baseline : tuple
    Tuple of ``(markerline, stemlines, baseline)``.
    ``markerline`` contains the `.Line2D` of the markers,
    ``stemlines`` is a `.LineCollection` of the main lines,
    ``baseline`` is the `.Line2D` of the baseline.
