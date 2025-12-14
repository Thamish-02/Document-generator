## AI Summary

A file named angle_helper.py.


### Function: select_step_degree(dv)

### Function: select_step_hour(dv)

### Function: select_step_sub(dv)

### Function: select_step(v1, v2, nv, hour, include_last, threshold_factor)

### Function: select_step24(v1, v2, nv, include_last, threshold_factor)

### Function: select_step360(v1, v2, nv, include_last, threshold_factor)

## Class: LocatorBase

## Class: LocatorHMS

## Class: LocatorHM

## Class: LocatorH

## Class: LocatorDMS

## Class: LocatorDM

## Class: LocatorD

## Class: FormatterDMS

## Class: FormatterHMS

## Class: ExtremeFinderCycle

### Function: __init__(self, nbins, include_last)

### Function: set_params(self, nbins)

### Function: __call__(self, v1, v2)

### Function: __call__(self, v1, v2)

### Function: __call__(self, v1, v2)

### Function: __call__(self, v1, v2)

### Function: __call__(self, v1, v2)

### Function: __call__(self, v1, v2)

### Function: _get_number_fraction(self, factor)

### Function: __call__(self, direction, factor, values)

### Function: __call__(self, direction, factor, values)

### Function: __init__(self, nx, ny, lon_cycle, lat_cycle, lon_minmax, lat_minmax)

**Description:** This subclass handles the case where one or both coordinates should be
taken modulo 360, or be restricted to not exceed a specific range.

Parameters
----------
nx, ny : int
    The number of samples in each direction.

lon_cycle, lat_cycle : 360 or None
    If not None, values in the corresponding direction are taken modulo
    *lon_cycle* or *lat_cycle*; in theory this can be any number but
    the implementation actually assumes that it is 360 (if not None);
    other values give nonsensical results.

    This is done by "unwrapping" the transformed grid coordinates so
    that jumps are less than a half-cycle; then normalizing the span to
    no more than a full cycle.

    For example, if values are in the union of the [0, 2] and
    [358, 360] intervals (typically, angles measured modulo 360), the
    values in the second interval are normalized to [-2, 0] instead so
    that the values now cover [-2, 2].  If values are in a range of
    [5, 1000], this gets normalized to [5, 365].

lon_minmax, lat_minmax : (float, float) or None
    If not None, the computed bounding box is clipped to the given
    range in the corresponding direction.

### Function: __call__(self, transform_xy, x1, y1, x2, y2)
