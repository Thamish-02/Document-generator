## AI Summary

A file named geometry.py.


### Function: _vector_between(origin, target)

### Function: _round_point(pt)

### Function: _unit_vector(vec)

### Function: _rounding_offset(direction)

## Class: Circle

### Function: round_start_circle_stable_containment(c0, r0, c1, r1)

**Description:** Round start circle so that it stays inside/outside end circle after rounding.

The rounding of circle coordinates to integers may cause an abrupt change
if the start circle c0 is so close to the end circle c1's perimiter that
it ends up falling outside (or inside) as a result of the rounding.
To keep the gradient unchanged, we nudge it in the right direction.

See:
https://github.com/googlefonts/colr-gradients-spec/issues/204
https://github.com/googlefonts/picosvg/issues/158

### Function: __init__(self, centre, radius)

### Function: __repr__(self)

### Function: round(self)

### Function: inside(self, outer_circle, tolerance)

### Function: concentric(self, other)

### Function: move(self, dx, dy)
