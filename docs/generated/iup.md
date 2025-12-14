## AI Summary

A file named iup.py.


### Function: iup_segment(coords, rc1, rd1, rc2, rd2)

**Description:** Given two reference coordinates `rc1` & `rc2` and their respective
delta vectors `rd1` & `rd2`, returns interpolated deltas for the set of
coordinates `coords`.

### Function: iup_contour(deltas, coords)

**Description:** For the contour given in `coords`, interpolate any missing
delta values in delta vector `deltas`.

Returns fully filled-out delta vector.

### Function: iup_delta(deltas, coords, ends)

**Description:** For the outline given in `coords`, with contour endpoints given
in sorted increasing order in `ends`, interpolate any missing
delta values in delta vector `deltas`.

Returns fully filled-out delta vector.

### Function: can_iup_in_between(deltas, coords, i, j, tolerance)

**Description:** Return true if the deltas for points at `i` and `j` (`i < j`) can be
successfully used to interpolate deltas for points in between them within
provided error tolerance.

### Function: _iup_contour_bound_forced_set(deltas, coords, tolerance)

**Description:** The forced set is a conservative set of points on the contour that must be encoded
explicitly (ie. cannot be interpolated).  Calculating this set allows for significantly
speeding up the dynamic-programming, as well as resolve circularity in DP.

The set is precise; that is, if an index is in the returned set, then there is no way
that IUP can generate delta for that point, given `coords` and `deltas`.

### Function: _iup_contour_optimize_dp(deltas, coords, forced, tolerance, lookback)

**Description:** Straightforward Dynamic-Programming.  For each index i, find least-costly encoding of
points 0 to i where i is explicitly encoded.  We find this by considering all previous
explicit points j and check whether interpolation can fill points between j and i.

Note that solution always encodes last point explicitly.  Higher-level is responsible
for removing that restriction.

As major speedup, we stop looking further whenever we see a "forced" point.

### Function: _rot_list(l, k)

**Description:** Rotate list by k items forward.  Ie. item at position 0 will be
at position k in returned list.  Negative k is allowed.

### Function: _rot_set(s, k, n)

### Function: iup_contour_optimize(deltas, coords, tolerance)

**Description:** For contour with coordinates `coords`, optimize a set of delta
values `deltas` within error `tolerance`.

Returns delta vector that has most number of None items instead of
the input delta.

### Function: iup_delta_optimize(deltas, coords, ends, tolerance)

**Description:** For the outline given in `coords`, with contour endpoints given
in sorted increasing order in `ends`, optimize a set of delta
values `deltas` within error `tolerance`.

Returns delta vector that has most number of None items instead of
the input delta.
