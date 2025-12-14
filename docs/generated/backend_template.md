## AI Summary

A file named backend_template.py.


## Class: RendererTemplate

**Description:** The renderer handles drawing/rendering operations.

This is a minimal do-nothing class that can be used to get started when
writing a new backend.  Refer to `.backend_bases.RendererBase` for
documentation of the methods.

## Class: GraphicsContextTemplate

**Description:** The graphics context provides the color, line styles, etc.  See the cairo
and postscript backends for examples of mapping the graphics context
attributes (cap styles, join styles, line widths, colors) to a particular
backend.  In cairo this is done by wrapping a cairo.Context object and
forwarding the appropriate calls to it using a dictionary mapping styles
to gdk constants.  In Postscript, all the work is done by the renderer,
mapping line styles to postscript calls.

If it's more appropriate to do the mapping at the renderer level (as in
the postscript backend), you don't need to override any of the GC methods.
If it's more appropriate to wrap an instance (as in the cairo backend) and
do the mapping here, you'll need to override several of the setter
methods.

The base GraphicsContext stores colors as an RGB tuple on the unit
interval, e.g., (0.5, 0.0, 1.0). You may need to map this to colors
appropriate for your backend.

## Class: FigureManagerTemplate

**Description:** Helper class for pyplot mode, wraps everything up into a neat bundle.

For non-interactive backends, the base class is sufficient.  For
interactive backends, see the documentation of the `.FigureManagerBase`
class for the list of methods that can/should be overridden.

## Class: FigureCanvasTemplate

**Description:** The canvas the figure renders into.  Calls the draw and print fig
methods, creates the renderers, etc.

Note: GUI templates will want to connect events for button presses,
mouse movements and key presses to functions that call the base
class methods button_press_event, button_release_event,
motion_notify_event, key_press_event, and key_release_event.  See the
implementations of the interactive backends for examples.

Attributes
----------
figure : `~matplotlib.figure.Figure`
    A high-level Figure instance

### Function: __init__(self, dpi)

### Function: draw_path(self, gc, path, transform, rgbFace)

### Function: draw_image(self, gc, x, y, im)

### Function: draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)

### Function: flipy(self)

### Function: get_canvas_width_height(self)

### Function: get_text_width_height_descent(self, s, prop, ismath)

### Function: new_gc(self)

### Function: points_to_pixels(self, points)

### Function: draw(self)

**Description:** Draw the figure using the renderer.

It is important that this method actually walk the artist tree
even if not output is produced because this will trigger
deferred work (like computing limits auto-limits and tick
values) that users may want access to before saving to disk.

### Function: print_foo(self, filename)

**Description:** Write out format foo.

This method is normally called via `.Figure.savefig` and
`.FigureCanvasBase.print_figure`, which take care of setting the figure
facecolor, edgecolor, and dpi to the desired output values, and will
restore them to the original values.  Therefore, `print_foo` does not
need to handle these settings.

### Function: get_default_filetype(self)
