## AI Summary

A file named axes_rgb.py.


### Function: make_rgb_axes(ax, pad, axes_class)

**Description:** Parameters
----------
ax : `~matplotlib.axes.Axes`
    Axes instance to create the RGB Axes in.
pad : float, optional
    Fraction of the Axes height to pad.
axes_class : `matplotlib.axes.Axes` or None, optional
    Axes class to use for the R, G, and B Axes. If None, use
    the same class as *ax*.
**kwargs
    Forwarded to *axes_class* init for the R, G, and B Axes.

## Class: RGBAxes

**Description:** 4-panel `~.Axes.imshow` (RGB, R, G, B).

Layout::

    ┌───────────────┬─────┐
    │               │  R  │
    │               ├─────┤
    │      RGB      │  G  │
    │               ├─────┤
    │               │  B  │
    └───────────────┴─────┘

Subclasses can override the ``_defaultAxesClass`` attribute.
By default RGBAxes uses `.mpl_axes.Axes`.

Attributes
----------
RGB : ``_defaultAxesClass``
    The Axes object for the three-channel `~.Axes.imshow`.
R : ``_defaultAxesClass``
    The Axes object for the red channel `~.Axes.imshow`.
G : ``_defaultAxesClass``
    The Axes object for the green channel `~.Axes.imshow`.
B : ``_defaultAxesClass``
    The Axes object for the blue channel `~.Axes.imshow`.

### Function: __init__(self)

**Description:** Parameters
----------
pad : float, default: 0
    Fraction of the Axes height to put as padding.
axes_class : `~matplotlib.axes.Axes`
    Axes class to use. If not provided, ``_defaultAxesClass`` is used.
*args
    Forwarded to *axes_class* init for the RGB Axes
**kwargs
    Forwarded to *axes_class* init for the RGB, R, G, and B Axes

### Function: imshow_rgb(self, r, g, b)

**Description:** Create the four images {rgb, r, g, b}.

Parameters
----------
r, g, b : array-like
    The red, green, and blue arrays.
**kwargs
    Forwarded to `~.Axes.imshow` calls for the four images.

Returns
-------
rgb : `~matplotlib.image.AxesImage`
r : `~matplotlib.image.AxesImage`
g : `~matplotlib.image.AxesImage`
b : `~matplotlib.image.AxesImage`
