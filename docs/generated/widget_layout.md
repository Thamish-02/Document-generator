## AI Summary

A file named widget_layout.py.


## Class: Layout

**Description:** Layout specification

Defines a layout that can be expressed using CSS.  Supports a subset of
https://developer.mozilla.org/en-US/docs/Web/CSS/Reference

When a property is also accessible via a shorthand property, we only
expose the shorthand.

For example:
- ``flex-grow``, ``flex-shrink`` and ``flex-basis`` are bound to ``flex``.
- ``flex-wrap`` and ``flex-direction`` are bound to ``flex-flow``.
- ``margin-[top/bottom/left/right]`` values are bound to ``margin``, etc.

## Class: LayoutTraitType

### Function: __init__(self)

### Function: _get_border(self)

**Description:** `border` property getter. Return the common value of all side
borders if they are identical. Otherwise return None.

### Function: _set_border(self, border)

**Description:** `border` property setter. Set all 4 sides to `border` string.

### Function: validate(self, obj, value)
