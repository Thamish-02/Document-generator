## AI Summary

A file named colorizer.py.


## Class: Colorizer

**Description:** Data to color pipeline.

This pipeline is accessible via `.Colorizer.to_rgba` and executed via
the `.Colorizer.norm` and `.Colorizer.cmap` attributes.

Parameters
----------
cmap: colorbar.Colorbar or str or None, default: None
    The colormap used to color data.

norm: colors.Normalize or str or None, default: None
    The normalization used to normalize the data

## Class: _ColorizerInterface

**Description:** Base class that contains the interface to `Colorizer` objects from
a `ColorizingArtist` or `.cm.ScalarMappable`.

Note: This class only contain functions that interface the .colorizer
attribute. Other functions that as shared between `.ColorizingArtist`
and `.cm.ScalarMappable` are not included.

## Class: _ScalarMappable

**Description:** A mixin class to map one or multiple sets of scalar data to RGBA.

The ScalarMappable applies data normalization before returning RGBA colors from
the given `~matplotlib.colors.Colormap`.

## Class: ColorizingArtist

**Description:** Base class for artists that make map data to color using a `.colorizer.Colorizer`.

The `.colorizer.Colorizer` applies data normalization before
returning RGBA colors from a `~matplotlib.colors.Colormap`.

### Function: _auto_norm_from_scale(scale_cls)

**Description:** Automatically generate a norm class from *scale_cls*.

This differs from `.colors.make_norm_from_scale` in the following points:

- This function is not a class decorator, but directly returns a norm class
  (as if decorating `.Normalize`).
- The scale is automatically constructed with ``nonpositive="mask"``, if it
  supports such a parameter, to work around the difference in defaults
  between standard scales (which use "clip") and norms (which use "mask").

Note that ``make_norm_from_scale`` caches the generated norm classes
(not the instances) and reuses them for later calls.  For example,
``type(_auto_norm_from_scale("log")) == LogNorm``.

### Function: __init__(self, cmap, norm)

### Function: _scale_norm(self, norm, vmin, vmax, A)

**Description:** Helper for initial scaling.

Used by public functions that create a ScalarMappable and support
parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
will take precedence over *vmin*, *vmax*.

Note that this method does not set the norm.

### Function: norm(self)

### Function: norm(self, norm)

### Function: to_rgba(self, x, alpha, bytes, norm)

**Description:** Return a normalized RGBA array corresponding to *x*.

In the normal case, *x* is a 1D or 2D sequence of scalars, and
the corresponding `~numpy.ndarray` of RGBA values will be returned,
based on the norm and colormap set for this Colorizer.

There is one special case, for handling images that are already
RGB or RGBA, such as might have been read from an image file.
If *x* is an `~numpy.ndarray` with 3 dimensions,
and the last dimension is either 3 or 4, then it will be
treated as an RGB or RGBA array, and no mapping will be done.
The array can be `~numpy.uint8`, or it can be floats with
values in the 0-1 range; otherwise a ValueError will be raised.
Any NaNs or masked elements will be set to 0 alpha.
If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
will be used to fill in the transparency.  If the last dimension
is 4, the *alpha* kwarg is ignored; it does not
replace the preexisting alpha.  A ValueError will be raised
if the third dimension is other than 3 or 4.

In either case, if *bytes* is *False* (default), the RGBA
array will be floats in the 0-1 range; if it is *True*,
the returned RGBA array will be `~numpy.uint8` in the 0 to 255 range.

If norm is False, no normalization of the input data is
performed, and it is assumed to be in the range (0-1).

### Function: _pass_image_data(x, alpha, bytes, norm)

**Description:** Helper function to pass ndarray of shape (...,3) or (..., 4)
through `to_rgba()`, see `to_rgba()` for docstring.

### Function: autoscale(self, A)

**Description:** Autoscale the scalar limits on the norm instance using the
current array

### Function: autoscale_None(self, A)

**Description:** Autoscale the scalar limits on the norm instance using the
current array, changing only limits that are None

### Function: _set_cmap(self, cmap)

**Description:** Set the colormap for luminance data.

Parameters
----------
cmap : `.Colormap` or str or None

### Function: cmap(self)

### Function: cmap(self, cmap)

### Function: set_clim(self, vmin, vmax)

**Description:** Set the norm limits for image scaling.

Parameters
----------
vmin, vmax : float
     The limits.

     The limits may also be passed as a tuple (*vmin*, *vmax*) as a
     single positional argument.

     .. ACCEPTS: (vmin: float, vmax: float)

### Function: get_clim(self)

**Description:** Return the values (min, max) that are mapped to the colormap limits.

### Function: changed(self)

**Description:** Call this whenever the mappable is changed to notify all the
callbackSM listeners to the 'changed' signal.

### Function: vmin(self)

### Function: vmin(self, vmin)

### Function: vmax(self)

### Function: vmax(self, vmax)

### Function: clip(self)

### Function: clip(self, clip)

### Function: _scale_norm(self, norm, vmin, vmax)

### Function: to_rgba(self, x, alpha, bytes, norm)

**Description:** Return a normalized RGBA array corresponding to *x*.

In the normal case, *x* is a 1D or 2D sequence of scalars, and
the corresponding `~numpy.ndarray` of RGBA values will be returned,
based on the norm and colormap set for this Colorizer.

There is one special case, for handling images that are already
RGB or RGBA, such as might have been read from an image file.
If *x* is an `~numpy.ndarray` with 3 dimensions,
and the last dimension is either 3 or 4, then it will be
treated as an RGB or RGBA array, and no mapping will be done.
The array can be `~numpy.uint8`, or it can be floats with
values in the 0-1 range; otherwise a ValueError will be raised.
Any NaNs or masked elements will be set to 0 alpha.
If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
will be used to fill in the transparency.  If the last dimension
is 4, the *alpha* kwarg is ignored; it does not
replace the preexisting alpha.  A ValueError will be raised
if the third dimension is other than 3 or 4.

In either case, if *bytes* is *False* (default), the RGBA
array will be floats in the 0-1 range; if it is *True*,
the returned RGBA array will be `~numpy.uint8` in the 0 to 255 range.

If norm is False, no normalization of the input data is
performed, and it is assumed to be in the range (0-1).

### Function: get_clim(self)

**Description:** Return the values (min, max) that are mapped to the colormap limits.

### Function: set_clim(self, vmin, vmax)

**Description:** Set the norm limits for image scaling.

Parameters
----------
vmin, vmax : float
     The limits.

     For scalar data, the limits may also be passed as a
     tuple (*vmin*, *vmax*) as a single positional argument.

     .. ACCEPTS: (vmin: float, vmax: float)

### Function: get_alpha(self)

### Function: cmap(self)

### Function: cmap(self, cmap)

### Function: get_cmap(self)

**Description:** Return the `.Colormap` instance.

### Function: set_cmap(self, cmap)

**Description:** Set the colormap for luminance data.

Parameters
----------
cmap : `.Colormap` or str or None

### Function: norm(self)

### Function: norm(self, norm)

### Function: set_norm(self, norm)

**Description:** Set the normalization instance.

Parameters
----------
norm : `.Normalize` or str or None

Notes
-----
If there are any colorbars using the mappable for this norm, setting
the norm of the mappable will reset the norm, locator, and formatters
on the colorbar to default.

### Function: autoscale(self)

**Description:** Autoscale the scalar limits on the norm instance using the
current array

### Function: autoscale_None(self)

**Description:** Autoscale the scalar limits on the norm instance using the
current array, changing only limits that are None

### Function: colorbar(self)

**Description:** The last colorbar associated with this object. May be None

### Function: colorbar(self, colorbar)

### Function: _format_cursor_data_override(self, data)

### Function: __init__(self, norm, cmap)

**Description:** Parameters
----------
norm : `.Normalize` (or subclass thereof) or str or None
    The normalizing object which scales data, typically into the
    interval ``[0, 1]``.
    If a `str`, a `.Normalize` subclass is dynamically generated based
    on the scale with the corresponding name.
    If *None*, *norm* defaults to a *colors.Normalize* object which
    initializes its scaling based on the first data processed.
cmap : str or `~matplotlib.colors.Colormap`
    The colormap used to map normalized data values to RGBA colors.

### Function: set_array(self, A)

**Description:** Set the value array from array-like *A*.

Parameters
----------
A : array-like or None
    The values that are mapped to colors.

    The base class `.ScalarMappable` does not make any assumptions on
    the dimensionality and shape of the value array *A*.

### Function: get_array(self)

**Description:** Return the array of values, that are mapped to colors.

The base class `.ScalarMappable` does not make any assumptions on
the dimensionality and shape of the array.

### Function: changed(self)

**Description:** Call this whenever the mappable is changed to notify all the
callbackSM listeners to the 'changed' signal.

### Function: _check_exclusionary_keywords(colorizer)

**Description:** Raises a ValueError if any kwarg is not None while colorizer is not None

### Function: _get_colorizer(cmap, norm, colorizer)

### Function: __init__(self, colorizer)

**Description:** Parameters
----------
colorizer : `.colorizer.Colorizer`

### Function: colorizer(self)

### Function: colorizer(self, cl)

### Function: _set_colorizer_check_keywords(self, colorizer)

**Description:** Raises a ValueError if any kwarg is not None while colorizer is not None.
