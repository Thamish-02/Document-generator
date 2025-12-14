## AI Summary

A file named interpolatableHelpers.py.


## Class: InterpolatableProblem

### Function: sort_problems(problems)

**Description:** Sort problems by severity, then by glyph name, then by problem message.

### Function: rot_list(l, k)

**Description:** Rotate list by k items forward.  Ie. item at position 0 will be
at position k in returned list.  Negative k is allowed.

## Class: PerContourPen

## Class: PerContourOrComponentPen

## Class: SimpleRecordingPointPen

### Function: vdiff_hypot2(v0, v1)

### Function: vdiff_hypot2_complex(v0, v1)

### Function: matching_cost(G, matching)

### Function: min_cost_perfect_bipartite_matching_scipy(G)

### Function: min_cost_perfect_bipartite_matching_munkres(G)

### Function: min_cost_perfect_bipartite_matching_bruteforce(G)

### Function: contour_vector_from_stats(stats)

### Function: matching_for_vectors(m0, m1)

### Function: points_characteristic_bits(points)

### Function: points_complex_vector(points)

### Function: add_isomorphisms(points, isomorphisms, reverse)

### Function: find_parents_and_order(glyphsets, locations)

### Function: transform_from_stats(stats, inverse)

### Function: __init__(self, Pen, glyphset)

### Function: _moveTo(self, p0)

### Function: _lineTo(self, p1)

### Function: _qCurveToOne(self, p1, p2)

### Function: _curveToOne(self, p1, p2, p3)

### Function: _closePath(self)

### Function: _endPath(self)

### Function: _newItem(self)

### Function: addComponent(self, glyphName, transformation)

### Function: __init__(self)

### Function: beginPath(self, identifier)

### Function: endPath(self)

### Function: addPoint(self, pt, segmentType)
