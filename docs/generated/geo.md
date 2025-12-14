## AI Summary

A file named geo.py.


## Class: GeoAxes

**Description:** An abstract base class for geographic projections.

## Class: _GeoTransform

## Class: AitoffAxes

## Class: HammerAxes

## Class: MollweideAxes

## Class: LambertAxes

## Class: ThetaFormatter

**Description:** Used to format the theta tick labels.  Converts the native
unit of radians into degrees and adds a degree symbol.

### Function: _init_axis(self)

### Function: clear(self)

### Function: _set_lim_and_transforms(self)

### Function: _get_affine_transform(self)

### Function: get_xaxis_transform(self, which)

### Function: get_xaxis_text1_transform(self, pad)

### Function: get_xaxis_text2_transform(self, pad)

### Function: get_yaxis_transform(self, which)

### Function: get_yaxis_text1_transform(self, pad)

### Function: get_yaxis_text2_transform(self, pad)

### Function: _gen_axes_patch(self)

### Function: _gen_axes_spines(self)

### Function: set_yscale(self)

### Function: set_xlim(self)

**Description:** Not supported. Please consider using Cartopy.

### Function: invert_xaxis(self)

**Description:** Not supported. Please consider using Cartopy.

### Function: format_coord(self, lon, lat)

**Description:** Return a format string formatting the coordinate.

### Function: set_longitude_grid(self, degrees)

**Description:** Set the number of degrees between each longitude grid.

### Function: set_latitude_grid(self, degrees)

**Description:** Set the number of degrees between each latitude grid.

### Function: set_longitude_grid_ends(self, degrees)

**Description:** Set the latitude(s) at which to stop drawing the longitude grids.

### Function: get_data_ratio(self)

**Description:** Return the aspect ratio of the data itself.

### Function: can_zoom(self)

**Description:** Return whether this Axes supports the zoom box button functionality.

This Axes object does not support interactive zoom box.

### Function: can_pan(self)

**Description:** Return whether this Axes supports the pan/zoom button functionality.

This Axes object does not support interactive pan/zoom.

### Function: start_pan(self, x, y, button)

### Function: end_pan(self)

### Function: drag_pan(self, button, key, x, y)

### Function: __init__(self, resolution)

**Description:** Create a new geographical transform.

Resolution is the number of steps to interpolate between each input
line segment to approximate its path in curved space.

### Function: __str__(self)

### Function: transform_path_non_affine(self, path)

## Class: AitoffTransform

**Description:** The base Aitoff transform.

## Class: InvertedAitoffTransform

### Function: __init__(self)

### Function: _get_core_transform(self, resolution)

## Class: HammerTransform

**Description:** The base Hammer transform.

## Class: InvertedHammerTransform

### Function: __init__(self)

### Function: _get_core_transform(self, resolution)

## Class: MollweideTransform

**Description:** The base Mollweide transform.

## Class: InvertedMollweideTransform

### Function: __init__(self)

### Function: _get_core_transform(self, resolution)

## Class: LambertTransform

**Description:** The base Lambert transform.

## Class: InvertedLambertTransform

### Function: __init__(self)

### Function: clear(self)

### Function: _get_core_transform(self, resolution)

### Function: _get_affine_transform(self)

### Function: __init__(self, round_to)

### Function: __call__(self, x, pos)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, center_longitude, center_latitude, resolution)

**Description:** Create a new Lambert transform.  Resolution is the number of steps
to interpolate between each input line segment to approximate its
path in curved Lambert space.

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, center_longitude, center_latitude, resolution)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: d(theta)
