## AI Summary

A file named backend_cairo.py.


### Function: _set_rgba(ctx, color, alpha, forced_alpha)

### Function: _append_path(ctx, path, transform, clip)

### Function: _cairo_font_args_from_font_prop(prop)

**Description:** Convert a `.FontProperties` or a `.FontEntry` to arguments that can be
passed to `.Context.select_font_face`.

## Class: RendererCairo

## Class: GraphicsContextCairo

## Class: _CairoRegion

## Class: FigureCanvasCairo

## Class: _BackendCairo

### Function: attr(field)

### Function: __init__(self, dpi)

### Function: set_context(self, ctx)

### Function: _fill_and_stroke(ctx, fill_c, alpha, alpha_overrides)

### Function: draw_path(self, gc, path, transform, rgbFace)

### Function: draw_markers(self, gc, marker_path, marker_trans, path, transform, rgbFace)

### Function: draw_image(self, gc, x, y, im)

### Function: draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)

### Function: _draw_mathtext(self, gc, x, y, s, prop, angle)

### Function: get_canvas_width_height(self)

### Function: get_text_width_height_descent(self, s, prop, ismath)

### Function: new_gc(self)

### Function: points_to_pixels(self, points)

### Function: __init__(self, renderer)

### Function: restore(self)

### Function: set_alpha(self, alpha)

### Function: set_antialiased(self, b)

### Function: get_antialiased(self)

### Function: set_capstyle(self, cs)

### Function: set_clip_rectangle(self, rectangle)

### Function: set_clip_path(self, path)

### Function: set_dashes(self, offset, dashes)

### Function: set_foreground(self, fg, isRGBA)

### Function: get_rgb(self)

### Function: set_joinstyle(self, js)

### Function: set_linewidth(self, w)

### Function: __init__(self, slices, data)

### Function: _renderer(self)

### Function: get_renderer(self)

### Function: copy_from_bbox(self, bbox)

### Function: restore_region(self, region)

### Function: print_png(self, fobj)

### Function: print_rgba(self, fobj)

### Function: _get_printed_image_surface(self)

### Function: _save(self, fmt, fobj)
