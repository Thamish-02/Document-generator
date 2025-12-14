## AI Summary

A file named patheffects.py.


## Class: AbstractPathEffect

**Description:** A base class for path effects.

Subclasses should override the ``draw_path`` method to add effect
functionality.

## Class: PathEffectRenderer

**Description:** Implements a Renderer which contains another renderer.

This proxy then intercepts draw calls, calling the appropriate
:class:`AbstractPathEffect` draw method.

.. note::
    Not all methods have been overridden on this RendererBase subclass.
    It may be necessary to add further methods to extend the PathEffects
    capabilities further.

## Class: Normal

**Description:** The "identity" PathEffect.

The Normal PathEffect's sole purpose is to draw the original artist with
no special path effect.

### Function: _subclass_with_normal(effect_class)

**Description:** Create a PathEffect class combining *effect_class* and a normal draw.

## Class: Stroke

**Description:** A line based PathEffect which re-draws a stroke.

## Class: SimplePatchShadow

**Description:** A simple shadow via a filled patch.

## Class: SimpleLineShadow

**Description:** A simple shadow via a line.

## Class: PathPatchEffect

**Description:** Draws a `.PathPatch` instance whose Path comes from the original
PathEffect artist.

## Class: TickedStroke

**Description:** A line-based PathEffect which draws a path with a ticked style.

This line style is frequently used to represent constraints in
optimization.  The ticks may be used to indicate that one side
of the line is invalid or to represent a closed boundary of a
domain (i.e. a wall or the edge of a pipe).

The spacing, length, and angle of ticks can be controlled.

This line style is sometimes referred to as a hatched line.

See also the :doc:`/gallery/misc/tickedstroke_demo` example.

### Function: __init__(self, offset)

**Description:** Parameters
----------
offset : (float, float), default: (0, 0)
    The (x, y) offset to apply to the path, measured in points.

### Function: _offset_transform(self, renderer)

**Description:** Apply the offset to the given transform.

### Function: _update_gc(self, gc, new_gc_dict)

**Description:** Update the given GraphicsContext with the given dict of properties.

The keys in the dictionary are used to identify the appropriate
``set_`` method on the *gc*.

### Function: draw_path(self, renderer, gc, tpath, affine, rgbFace)

**Description:** Derived should override this method. The arguments are the same
as :meth:`matplotlib.backend_bases.RendererBase.draw_path`
except the first argument is a renderer.

### Function: __init__(self, path_effects, renderer)

**Description:** Parameters
----------
path_effects : iterable of :class:`AbstractPathEffect`
    The path effects which this renderer represents.
renderer : `~matplotlib.backend_bases.RendererBase` subclass

### Function: copy_with_path_effect(self, path_effects)

### Function: __getattribute__(self, name)

### Function: draw_path(self, gc, tpath, affine, rgbFace)

### Function: draw_markers(self, gc, marker_path, marker_trans, path)

### Function: draw_path_collection(self, gc, master_transform, paths)

### Function: open_group(self, s, gid)

### Function: close_group(self, s)

## Class: withEffect

### Function: __init__(self, offset)

**Description:** The path will be stroked with its gc updated with the given
keyword arguments, i.e., the keyword arguments should be valid
gc parameter values.

### Function: draw_path(self, renderer, gc, tpath, affine, rgbFace)

**Description:** Draw the path with updated gc.

### Function: __init__(self, offset, shadow_rgbFace, alpha, rho)

**Description:** Parameters
----------
offset : (float, float), default: (2, -2)
    The (x, y) offset of the shadow in points.
shadow_rgbFace : :mpltype:`color`
    The shadow color.
alpha : float, default: 0.3
    The alpha transparency of the created shadow patch.
rho : float, default: 0.3
    A scale factor to apply to the rgbFace color if *shadow_rgbFace*
    is not specified.
**kwargs
    Extra keywords are stored and passed through to
    :meth:`AbstractPathEffect._update_gc`.

### Function: draw_path(self, renderer, gc, tpath, affine, rgbFace)

**Description:** Overrides the standard draw_path to add the shadow offset and
necessary color changes for the shadow.

### Function: __init__(self, offset, shadow_color, alpha, rho)

**Description:** Parameters
----------
offset : (float, float), default: (2, -2)
    The (x, y) offset to apply to the path, in points.
shadow_color : :mpltype:`color`, default: 'black'
    The shadow color.
    A value of ``None`` takes the original artist's color
    with a scale factor of *rho*.
alpha : float, default: 0.3
    The alpha transparency of the created shadow patch.
rho : float, default: 0.3
    A scale factor to apply to the rgbFace color if *shadow_color*
    is ``None``.
**kwargs
    Extra keywords are stored and passed through to
    :meth:`AbstractPathEffect._update_gc`.

### Function: draw_path(self, renderer, gc, tpath, affine, rgbFace)

**Description:** Overrides the standard draw_path to add the shadow offset and
necessary color changes for the shadow.

### Function: __init__(self, offset)

**Description:** Parameters
----------
offset : (float, float), default: (0, 0)
    The (x, y) offset to apply to the path, in points.
**kwargs
    All keyword arguments are passed through to the
    :class:`~matplotlib.patches.PathPatch` constructor. The
    properties which cannot be overridden are "path", "clip_box"
    "transform" and "clip_path".

### Function: draw_path(self, renderer, gc, tpath, affine, rgbFace)

### Function: __init__(self, offset, spacing, angle, length)

**Description:** Parameters
----------
offset : (float, float), default: (0, 0)
    The (x, y) offset to apply to the path, in points.
spacing : float, default: 10.0
    The spacing between ticks in points.
angle : float, default: 45.0
    The angle between the path and the tick in degrees.  The angle
    is measured as if you were an ant walking along the curve, with
    zero degrees pointing directly ahead, 90 to your left, -90
    to your right, and 180 behind you. To change side of the ticks,
    change sign of the angle.
length : float, default: 1.414
    The length of the tick relative to spacing.
    Recommended length = 1.414 (sqrt(2)) when angle=45, length=1.0
    when angle=90 and length=2.0 when angle=60.
**kwargs
    Extra keywords are stored and passed through to
    :meth:`AbstractPathEffect._update_gc`.

Examples
--------
See :doc:`/gallery/misc/tickedstroke_demo`.

### Function: draw_path(self, renderer, gc, tpath, affine, rgbFace)

**Description:** Draw the path with updated gc.

### Function: draw_path(self, renderer, gc, tpath, affine, rgbFace)
