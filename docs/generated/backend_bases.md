## AI Summary

A file named backend_bases.py.


### Function: register_backend(format, backend, description)

**Description:** Register a backend for saving to a given file format.

Parameters
----------
format : str
    File extension
backend : module string or canvas class
    Backend for handling file output
description : str, default: ""
    Description of the file type.

### Function: get_registered_canvas_class(format)

**Description:** Return the registered default canvas for given file format.
Handles deferred import of required backend.

## Class: RendererBase

**Description:** An abstract base class to handle drawing/rendering operations.

The following methods must be implemented in the backend for full
functionality (though just implementing `draw_path` alone would give a
highly capable backend):

* `draw_path`
* `draw_image`
* `draw_gouraud_triangles`

The following methods *should* be implemented in the backend for
optimization reasons:

* `draw_text`
* `draw_markers`
* `draw_path_collection`
* `draw_quad_mesh`

## Class: GraphicsContextBase

**Description:** An abstract base class that provides color, line styles, etc.

## Class: TimerBase

**Description:** A base class for providing timer events, useful for things animations.
Backends need to implement a few specific methods in order to use their
own timing mechanisms so that the timer events are integrated into their
event loops.

Subclasses must override the following methods:

- ``_timer_start``: Backend-specific code for starting the timer.
- ``_timer_stop``: Backend-specific code for stopping the timer.

Subclasses may additionally override the following methods:

- ``_timer_set_single_shot``: Code for setting the timer to single shot
  operating mode, if supported by the timer object.  If not, the `Timer`
  class itself will store the flag and the ``_on_timer`` method should be
  overridden to support such behavior.

- ``_timer_set_interval``: Code for setting the interval on the timer, if
  there is a method for doing so on the timer object.

- ``_on_timer``: The internal function that any timer object should call,
  which will handle the task of running all callbacks that have been set.

## Class: Event

**Description:** A Matplotlib event.

The following attributes are defined and shown with their default values.
Subclasses may define additional attributes.

Attributes
----------
name : str
    The event name.
canvas : `FigureCanvasBase`
    The backend-specific canvas instance generating the event.
guiEvent
    The GUI event that triggered the Matplotlib event.

## Class: DrawEvent

**Description:** An event triggered by a draw operation on the canvas.

In most backends, callbacks subscribed to this event will be fired after
the rendering is complete but before the screen is updated. Any extra
artists drawn to the canvas's renderer will be reflected without an
explicit call to ``blit``.

.. warning::

   Calling ``canvas.draw`` and ``canvas.blit`` in these callbacks may
   not be safe with all backends and may cause infinite recursion.

A DrawEvent has a number of special attributes in addition to those defined
by the parent `Event` class.

Attributes
----------
renderer : `RendererBase`
    The renderer for the draw event.

## Class: ResizeEvent

**Description:** An event triggered by a canvas resize.

A ResizeEvent has a number of special attributes in addition to those
defined by the parent `Event` class.

Attributes
----------
width : int
    Width of the canvas in pixels.
height : int
    Height of the canvas in pixels.

## Class: CloseEvent

**Description:** An event triggered by a figure being closed.

## Class: LocationEvent

**Description:** An event that has a screen location.

A LocationEvent has a number of special attributes in addition to those
defined by the parent `Event` class.

Attributes
----------
x, y : int or None
    Event location in pixels from bottom left of canvas.
inaxes : `~matplotlib.axes.Axes` or None
    The `~.axes.Axes` instance over which the mouse is, if any.
xdata, ydata : float or None
    Data coordinates of the mouse within *inaxes*, or *None* if the mouse
    is not over an Axes.
modifiers : frozenset
    The keyboard modifiers currently being pressed (except for KeyEvent).

## Class: MouseButton

## Class: MouseEvent

**Description:** A mouse event ('button_press_event', 'button_release_event', 'scroll_event', 'motion_notify_event').

A MouseEvent has a number of special attributes in addition to those
defined by the parent `Event` and `LocationEvent` classes.

Attributes
----------
button : None or `MouseButton` or {'up', 'down'}
    The button pressed. 'up' and 'down' are used for scroll events.

    Note that LEFT and RIGHT actually refer to the "primary" and
    "secondary" buttons, i.e. if the user inverts their left and right
    buttons ("left-handed setting") then the LEFT button will be the one
    physically on the right.

    If this is unset, *name* is "scroll_event", and *step* is nonzero, then
    this will be set to "up" or "down" depending on the sign of *step*.

buttons : None or frozenset
    For 'motion_notify_event', the mouse buttons currently being pressed
    (a set of zero or more MouseButtons);
    for other events, None.

    .. note::
       For 'motion_notify_event', this attribute is more accurate than
       the ``button`` (singular) attribute, which is obtained from the last
       'button_press_event' or 'button_release_event' that occurred within
       the canvas (and thus 1. be wrong if the last change in mouse state
       occurred when the canvas did not have focus, and 2. cannot report
       when multiple buttons are pressed).

       This attribute is not set for 'button_press_event' and
       'button_release_event' because GUI toolkits are inconsistent as to
       whether they report the button state *before* or *after* the
       press/release occurred.

    .. warning::
       On macOS, the Tk backends only report a single button even if
       multiple buttons are pressed.

key : None or str
    The key pressed when the mouse event triggered, e.g. 'shift'.
    See `KeyEvent`.

    .. warning::
       This key is currently obtained from the last 'key_press_event' or
       'key_release_event' that occurred within the canvas.  Thus, if the
       last change of keyboard state occurred while the canvas did not have
       focus, this attribute will be wrong.  On the other hand, the
       ``modifiers`` attribute should always be correct, but it can only
       report on modifier keys.

step : float
    The number of scroll steps (positive for 'up', negative for 'down').
    This applies only to 'scroll_event' and defaults to 0 otherwise.

dblclick : bool
    Whether the event is a double-click. This applies only to
    'button_press_event' and is False otherwise. In particular, it's
    not used in 'button_release_event'.

Examples
--------
::

    def on_press(event):
        print('you pressed', event.button, event.xdata, event.ydata)

    cid = fig.canvas.mpl_connect('button_press_event', on_press)

## Class: PickEvent

**Description:** A pick event.

This event is fired when the user picks a location on the canvas
sufficiently close to an artist that has been made pickable with
`.Artist.set_picker`.

A PickEvent has a number of special attributes in addition to those defined
by the parent `Event` class.

Attributes
----------
mouseevent : `MouseEvent`
    The mouse event that generated the pick.
artist : `~matplotlib.artist.Artist`
    The picked artist.  Note that artists are not pickable by default
    (see `.Artist.set_picker`).
other
    Additional attributes may be present depending on the type of the
    picked object; e.g., a `.Line2D` pick may define different extra
    attributes than a `.PatchCollection` pick.

Examples
--------
Bind a function ``on_pick()`` to pick events, that prints the coordinates
of the picked data point::

    ax.plot(np.rand(100), 'o', picker=5)  # 5 points tolerance

    def on_pick(event):
        line = event.artist
        xdata, ydata = line.get_data()
        ind = event.ind
        print(f'on pick line: {xdata[ind]:.3f}, {ydata[ind]:.3f}')

    cid = fig.canvas.mpl_connect('pick_event', on_pick)

## Class: KeyEvent

**Description:** A key event (key press, key release).

A KeyEvent has a number of special attributes in addition to those defined
by the parent `Event` and `LocationEvent` classes.

Attributes
----------
key : None or str
    The key(s) pressed. Could be *None*, a single case sensitive Unicode
    character ("g", "G", "#", etc.), a special key ("control", "shift",
    "f1", "up", etc.) or a combination of the above (e.g., "ctrl+alt+g",
    "ctrl+alt+G").

Notes
-----
Modifier keys will be prefixed to the pressed key and will be in the order
"ctrl", "alt", "super". The exception to this rule is when the pressed key
is itself a modifier key, therefore "ctrl+alt" and "alt+control" can both
be valid key values.

Examples
--------
::

    def on_key(event):
        print('you pressed', event.key, event.xdata, event.ydata)

    cid = fig.canvas.mpl_connect('key_press_event', on_key)

### Function: _key_handler(event)

### Function: _mouse_handler(event)

### Function: _get_renderer(figure, print_method)

**Description:** Get the renderer that would be used to save a `.Figure`.

If you need a renderer without any active draw methods use
renderer._draw_disabled to temporary patch them out at your call site.

### Function: _no_output_draw(figure)

### Function: _is_non_interactive_terminal_ipython(ip)

**Description:** Return whether we are in a terminal IPython, but non interactive.

When in _terminal_ IPython, ip.parent will have and `interact` attribute,
if this attribute is False we do not setup eventloop integration as the
user will _not_ interact with IPython. In all other case (ZMQKernel, or is
interactive), we do.

### Function: _allow_interrupt(prepare_notifier, handle_sigint)

**Description:** A context manager that allows terminating a plot by sending a SIGINT.  It
is necessary because the running backend prevents the Python interpreter
from running and processing signals (i.e., to raise a KeyboardInterrupt).
To solve this, one needs to somehow wake up the interpreter and make it
close the plot window.  We do this by using the signal.set_wakeup_fd()
function which organizes a write of the signal number into a socketpair.
A backend-specific function, *prepare_notifier*, arranges to listen to
the pair's read socket while the event loop is running.  (If it returns a
notifier object, that object is kept alive while the context manager runs.)

If SIGINT was indeed caught, after exiting the on_signal() function the
interpreter reacts to the signal according to the handler function which
had been set up by a signal.signal() call; here, we arrange to call the
backend-specific *handle_sigint* function, passing the notifier object
as returned by prepare_notifier().  Finally, we call the old SIGINT
handler with the same arguments that were given to our custom handler.

We do this only if the old handler for SIGINT was not None, which means
that a non-python handler was installed, i.e. in Julia, and not SIG_IGN
which means we should ignore the interrupts.

Parameters
----------
prepare_notifier : Callable[[socket.socket], object]
handle_sigint : Callable[[object], object]

## Class: FigureCanvasBase

**Description:** The canvas the figure renders into.

Attributes
----------
figure : `~matplotlib.figure.Figure`
    A high-level figure instance.

### Function: key_press_handler(event, canvas, toolbar)

**Description:** Implement the default Matplotlib key bindings for the canvas and toolbar
described at :ref:`key-event-handling`.

Parameters
----------
event : `KeyEvent`
    A key press/release event.
canvas : `FigureCanvasBase`, default: ``event.canvas``
    The backend-specific canvas instance.  This parameter is kept for
    back-compatibility, but, if set, should always be equal to
    ``event.canvas``.
toolbar : `NavigationToolbar2`, default: ``event.canvas.toolbar``
    The navigation cursor toolbar.  This parameter is kept for
    back-compatibility, but, if set, should always be equal to
    ``event.canvas.toolbar``.

### Function: button_press_handler(event, canvas, toolbar)

**Description:** The default Matplotlib button actions for extra mouse buttons.

Parameters are as for `key_press_handler`, except that *event* is a
`MouseEvent`.

## Class: NonGuiException

**Description:** Raised when trying show a figure in a non-GUI backend.

## Class: FigureManagerBase

**Description:** A backend-independent abstraction of a figure container and controller.

The figure manager is used by pyplot to interact with the window in a
backend-independent way. It's an adapter for the real (GUI) framework that
represents the visual figure on screen.

The figure manager is connected to a specific canvas instance, which in turn
is connected to a specific figure instance. To access a figure manager for
a given figure in user code, you typically use ``fig.canvas.manager``.

GUI backends derive from this class to translate common operations such
as *show* or *resize* to the GUI-specific code. Non-GUI backends do not
support these operations and can just use the base class.

This following basic operations are accessible:

**Window operations**

- `~.FigureManagerBase.show`
- `~.FigureManagerBase.destroy`
- `~.FigureManagerBase.full_screen_toggle`
- `~.FigureManagerBase.resize`
- `~.FigureManagerBase.get_window_title`
- `~.FigureManagerBase.set_window_title`

**Key and mouse button press handling**

The figure manager sets up default key and mouse button press handling by
hooking up the `.key_press_handler` to the matplotlib event system. This
ensures the same shortcuts and mouse actions across backends.

**Other operations**

Subclasses will have additional attributes and functions to access
additional functionality. This is of course backend-specific. For example,
most GUI backends have ``window`` and ``toolbar`` attributes that give
access to the native GUI widgets of the respective framework.

Attributes
----------
canvas : `FigureCanvasBase`
    The backend-specific canvas instance.

num : int or str
    The figure number.

key_press_handler_id : int
    The default key handler cid, when using the toolmanager.
    To disable the default key press handling use::

        figure.canvas.mpl_disconnect(
            figure.canvas.manager.key_press_handler_id)

button_press_handler_id : int
    The default mouse button handler cid, when using the toolmanager.
    To disable the default button press handling use::

        figure.canvas.mpl_disconnect(
            figure.canvas.manager.button_press_handler_id)

## Class: _Mode

## Class: NavigationToolbar2

**Description:** Base class for the navigation cursor, version 2.

Backends must implement a canvas that handles connections for
'button_press_event' and 'button_release_event'.  See
:meth:`FigureCanvasBase.mpl_connect` for more information.

They must also define

:meth:`save_figure`
    Save the current figure.

:meth:`draw_rubberband` (optional)
    Draw the zoom to rect "rubberband" rectangle.

:meth:`set_message` (optional)
    Display message.

:meth:`set_history_buttons` (optional)
    You can change the history back / forward buttons to indicate disabled / enabled
    state.

and override ``__init__`` to set up the toolbar -- without forgetting to
call the base-class init.  Typically, ``__init__`` needs to set up toolbar
buttons connected to the `home`, `back`, `forward`, `pan`, `zoom`, and
`save_figure` methods and using standard icons in the "images" subdirectory
of the data path.

That's it, we'll do the rest!

## Class: ToolContainerBase

**Description:** Base class for all tool containers, e.g. toolbars.

Attributes
----------
toolmanager : `.ToolManager`
    The tools with which this `ToolContainer` wants to communicate.

## Class: _Backend

## Class: ShowBase

**Description:** Simple base class to generate a ``show()`` function in backends.

Subclass must override ``mainloop()`` method.

### Function: __init__(self)

### Function: open_group(self, s, gid)

**Description:** Open a grouping element with label *s* and *gid* (if set) as id.

Only used by the SVG renderer.

### Function: close_group(self, s)

**Description:** Close a grouping element with label *s*.

Only used by the SVG renderer.

### Function: draw_path(self, gc, path, transform, rgbFace)

**Description:** Draw a `~.path.Path` instance using the given affine transform.

### Function: draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace)

**Description:** Draw a marker at each of *path*'s vertices (excluding control points).

The base (fallback) implementation makes multiple calls to `draw_path`.
Backends may want to override this method in order to draw the marker
only once and reuse it multiple times.

Parameters
----------
gc : `.GraphicsContextBase`
    The graphics context.
marker_path : `~matplotlib.path.Path`
    The path for the marker.
marker_trans : `~matplotlib.transforms.Transform`
    An affine transform applied to the marker.
path : `~matplotlib.path.Path`
    The locations to draw the markers.
trans : `~matplotlib.transforms.Transform`
    An affine transform applied to the path.
rgbFace : :mpltype:`color`, optional

### Function: draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position)

**Description:** Draw a collection of *paths*.

Each path is first transformed by the corresponding entry
in *all_transforms* (a list of (3, 3) matrices) and then by
*master_transform*.  They are then translated by the corresponding
entry in *offsets*, which has been first transformed by *offset_trans*.

*facecolors*, *edgecolors*, *linewidths*, *linestyles*, and
*antialiased* are lists that set the corresponding properties.

*offset_position* is unused now, but the argument is kept for
backwards compatibility.

The base (fallback) implementation makes multiple calls to `draw_path`.
Backends may want to override this in order to render each set of
path data only once, and then reference that path multiple times with
the different offsets, colors, styles etc.  The generator methods
`_iter_collection_raw_paths` and `_iter_collection` are provided to
help with (and standardize) the implementation across backends.  It
is highly recommended to use those generators, so that changes to the
behavior of `draw_path_collection` can be made globally.

### Function: draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight, coordinates, offsets, offsetTrans, facecolors, antialiased, edgecolors)

**Description:** Draw a quadmesh.

The base (fallback) implementation converts the quadmesh to paths and
then calls `draw_path_collection`.

### Function: draw_gouraud_triangles(self, gc, triangles_array, colors_array, transform)

**Description:** Draw a series of Gouraud triangles.

Parameters
----------
gc : `.GraphicsContextBase`
    The graphics context.
triangles_array : (N, 3, 2) array-like
    Array of *N* (x, y) points for the triangles.
colors_array : (N, 3, 4) array-like
    Array of *N* RGBA colors for each point of the triangles.
transform : `~matplotlib.transforms.Transform`
    An affine transform to apply to the points.

### Function: _iter_collection_raw_paths(self, master_transform, paths, all_transforms)

**Description:** Helper method (along with `_iter_collection`) to implement
`draw_path_collection` in a memory-efficient manner.

This method yields all of the base path/transform combinations, given a
master transform, a list of paths and list of transforms.

The arguments should be exactly what is passed in to
`draw_path_collection`.

The backend should take each yielded path and transform and create an
object that can be referenced (reused) later.

### Function: _iter_collection_uses_per_path(self, paths, all_transforms, offsets, facecolors, edgecolors)

**Description:** Compute how many times each raw path object returned by
`_iter_collection_raw_paths` would be used when calling
`_iter_collection`. This is intended for the backend to decide
on the tradeoff between using the paths in-line and storing
them once and reusing. Rounds up in case the number of uses
is not the same for every path.

### Function: _iter_collection(self, gc, path_ids, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position)

**Description:** Helper method (along with `_iter_collection_raw_paths`) to implement
`draw_path_collection` in a memory-efficient manner.

This method yields all of the path, offset and graphics context
combinations to draw the path collection.  The caller should already
have looped over the results of `_iter_collection_raw_paths` to draw
this collection.

The arguments should be the same as that passed into
`draw_path_collection`, with the exception of *path_ids*, which is a
list of arbitrary objects that the backend will use to reference one of
the paths created in the `_iter_collection_raw_paths` stage.

Each yielded result is of the form::

   xo, yo, path_id, gc, rgbFace

where *xo*, *yo* is an offset; *path_id* is one of the elements of
*path_ids*; *gc* is a graphics context and *rgbFace* is a color to
use for filling the path.

### Function: get_image_magnification(self)

**Description:** Get the factor by which to magnify images passed to `draw_image`.
Allows a backend to have images at a different resolution to other
artists.

### Function: draw_image(self, gc, x, y, im, transform)

**Description:** Draw an RGBA image.

Parameters
----------
gc : `.GraphicsContextBase`
    A graphics context with clipping information.

x : float
    The distance in physical units (i.e., dots or pixels) from the left
    hand side of the canvas.

y : float
    The distance in physical units (i.e., dots or pixels) from the
    bottom side of the canvas.

im : (N, M, 4) array of `numpy.uint8`
    An array of RGBA pixels.

transform : `~matplotlib.transforms.Affine2DBase`
    If and only if the concrete backend is written such that
    `option_scale_image` returns ``True``, an affine transformation
    (i.e., an `.Affine2DBase`) *may* be passed to `draw_image`.  The
    translation vector of the transformation is given in physical units
    (i.e., dots or pixels). Note that the transformation does not
    override *x* and *y*, and has to be applied *before* translating
    the result by *x* and *y* (this can be accomplished by adding *x*
    and *y* to the translation vector defined by *transform*).

### Function: option_image_nocomposite(self)

**Description:** Return whether image composition by Matplotlib should be skipped.

Raster backends should usually return False (letting the C-level
rasterizer take care of image composition); vector backends should
usually return ``not rcParams["image.composite_image"]``.

### Function: option_scale_image(self)

**Description:** Return whether arbitrary affine transformations in `draw_image` are
supported (True for most vector backends).

### Function: draw_tex(self, gc, x, y, s, prop, angle)

**Description:** Draw a TeX instance.

Parameters
----------
gc : `.GraphicsContextBase`
    The graphics context.
x : float
    The x location of the text in display coords.
y : float
    The y location of the text baseline in display coords.
s : str
    The TeX text string.
prop : `~matplotlib.font_manager.FontProperties`
    The font properties.
angle : float
    The rotation angle in degrees anti-clockwise.
mtext : `~matplotlib.text.Text`
    The original text object to be rendered.

### Function: draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)

**Description:** Draw a text instance.

Parameters
----------
gc : `.GraphicsContextBase`
    The graphics context.
x : float
    The x location of the text in display coords.
y : float
    The y location of the text baseline in display coords.
s : str
    The text string.
prop : `~matplotlib.font_manager.FontProperties`
    The font properties.
angle : float
    The rotation angle in degrees anti-clockwise.
ismath : bool or "TeX"
    If True, use mathtext parser.
mtext : `~matplotlib.text.Text`
    The original text object to be rendered.

Notes
-----
**Notes for backend implementers:**

`.RendererBase.draw_text` also supports passing "TeX" to the *ismath*
parameter to use TeX rendering, but this is not required for actual
rendering backends, and indeed many builtin backends do not support
this.  Rather, TeX rendering is provided by `~.RendererBase.draw_tex`.

### Function: _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath)

**Description:** Draw the text by converting them to paths using `.TextToPath`.

This private helper supports the same parameters as
`~.RendererBase.draw_text`; setting *ismath* to "TeX" triggers TeX
rendering.

### Function: get_text_width_height_descent(self, s, prop, ismath)

**Description:** Get the width, height, and descent (offset from the bottom to the baseline), in
display coords, of the string *s* with `.FontProperties` *prop*.

Whitespace at the start and the end of *s* is included in the reported width.

### Function: flipy(self)

**Description:** Return whether y values increase from top to bottom.

Note that this only affects drawing of texts.

### Function: get_canvas_width_height(self)

**Description:** Return the canvas width and height in display coords.

### Function: get_texmanager(self)

**Description:** Return the `.TexManager` instance.

### Function: new_gc(self)

**Description:** Return an instance of a `.GraphicsContextBase`.

### Function: points_to_pixels(self, points)

**Description:** Convert points to display units.

You need to override this function (unless your backend
doesn't have a dpi, e.g., postscript or svg).  Some imaging
systems assume some value for pixels per inch::

    points to pixels = points * pixels_per_inch/72 * dpi/72

Parameters
----------
points : float or array-like

Returns
-------
Points converted to pixels

### Function: start_rasterizing(self)

**Description:** Switch to the raster renderer.

Used by `.MixedModeRenderer`.

### Function: stop_rasterizing(self)

**Description:** Switch back to the vector renderer and draw the contents of the raster
renderer as an image on the vector renderer.

Used by `.MixedModeRenderer`.

### Function: start_filter(self)

**Description:** Switch to a temporary renderer for image filtering effects.

Currently only supported by the agg renderer.

### Function: stop_filter(self, filter_func)

**Description:** Switch back to the original renderer.  The contents of the temporary
renderer is processed with the *filter_func* and is drawn on the
original renderer as an image.

Currently only supported by the agg renderer.

### Function: _draw_disabled(self)

**Description:** Context manager to temporary disable drawing.

This is used for getting the drawn size of Artists.  This lets us
run the draw process to update any Python state but does not pay the
cost of the draw_XYZ calls on the canvas.

### Function: __init__(self)

### Function: copy_properties(self, gc)

**Description:** Copy properties from *gc* to self.

### Function: restore(self)

**Description:** Restore the graphics context from the stack - needed only
for backends that save graphics contexts on a stack.

### Function: get_alpha(self)

**Description:** Return the alpha value used for blending - not supported on all
backends.

### Function: get_antialiased(self)

**Description:** Return whether the object should try to do antialiased rendering.

### Function: get_capstyle(self)

**Description:** Return the `.CapStyle`.

### Function: get_clip_rectangle(self)

**Description:** Return the clip rectangle as a `~matplotlib.transforms.Bbox` instance.

### Function: get_clip_path(self)

**Description:** Return the clip path in the form (path, transform), where path
is a `~.path.Path` instance, and transform is
an affine transform to apply to the path before clipping.

### Function: get_dashes(self)

**Description:** Return the dash style as an (offset, dash-list) pair.

See `.set_dashes` for details.

Default value is (None, None).

### Function: get_forced_alpha(self)

**Description:** Return whether the value given by get_alpha() should be used to
override any other alpha-channel values.

### Function: get_joinstyle(self)

**Description:** Return the `.JoinStyle`.

### Function: get_linewidth(self)

**Description:** Return the line width in points.

### Function: get_rgb(self)

**Description:** Return a tuple of three or four floats from 0-1.

### Function: get_url(self)

**Description:** Return a url if one is set, None otherwise.

### Function: get_gid(self)

**Description:** Return the object identifier if one is set, None otherwise.

### Function: get_snap(self)

**Description:** Return the snap setting, which can be:

* True: snap vertices to the nearest pixel center
* False: leave vertices as-is
* None: (auto) If the path contains only rectilinear line segments,
  round to the nearest pixel center

### Function: set_alpha(self, alpha)

**Description:** Set the alpha value used for blending - not supported on all backends.

If ``alpha=None`` (the default), the alpha components of the
foreground and fill colors will be used to set their respective
transparencies (where applicable); otherwise, ``alpha`` will override
them.

### Function: set_antialiased(self, b)

**Description:** Set whether object should be drawn with antialiased rendering.

### Function: set_capstyle(self, cs)

**Description:** Set how to draw endpoints of lines.

Parameters
----------
cs : `.CapStyle` or %(CapStyle)s

### Function: set_clip_rectangle(self, rectangle)

**Description:** Set the clip rectangle to a `.Bbox` or None.

### Function: set_clip_path(self, path)

**Description:** Set the clip path to a `.TransformedPath` or None.

### Function: set_dashes(self, dash_offset, dash_list)

**Description:** Set the dash style for the gc.

Parameters
----------
dash_offset : float
    Distance, in points, into the dash pattern at which to
    start the pattern. It is usually set to 0.
dash_list : array-like or None
    The on-off sequence as points.  None specifies a solid line. All
    values must otherwise be non-negative (:math:`\ge 0`).

Notes
-----
See p. 666 of the PostScript
`Language Reference
<https://www.adobe.com/jp/print/postscript/pdfs/PLRM.pdf>`_
for more info.

### Function: set_foreground(self, fg, isRGBA)

**Description:** Set the foreground color.

Parameters
----------
fg : :mpltype:`color`
isRGBA : bool
    If *fg* is known to be an ``(r, g, b, a)`` tuple, *isRGBA* can be
    set to True to improve performance.

### Function: set_joinstyle(self, js)

**Description:** Set how to draw connections between line segments.

Parameters
----------
js : `.JoinStyle` or %(JoinStyle)s

### Function: set_linewidth(self, w)

**Description:** Set the linewidth in points.

### Function: set_url(self, url)

**Description:** Set the url for links in compatible backends.

### Function: set_gid(self, id)

**Description:** Set the id.

### Function: set_snap(self, snap)

**Description:** Set the snap setting which may be:

* True: snap vertices to the nearest pixel center
* False: leave vertices as-is
* None: (auto) If the path contains only rectilinear line segments,
  round to the nearest pixel center

### Function: set_hatch(self, hatch)

**Description:** Set the hatch style (for fills).

### Function: get_hatch(self)

**Description:** Get the current hatch style.

### Function: get_hatch_path(self, density)

**Description:** Return a `.Path` for the current hatch.

### Function: get_hatch_color(self)

**Description:** Get the hatch color.

### Function: set_hatch_color(self, hatch_color)

**Description:** Set the hatch color.

### Function: get_hatch_linewidth(self)

**Description:** Get the hatch linewidth.

### Function: set_hatch_linewidth(self, hatch_linewidth)

**Description:** Set the hatch linewidth.

### Function: get_sketch_params(self)

**Description:** Return the sketch parameters for the artist.

Returns
-------
tuple or `None`

    A 3-tuple with the following elements:

    * ``scale``: The amplitude of the wiggle perpendicular to the
      source line.
    * ``length``: The length of the wiggle along the line.
    * ``randomness``: The scale factor by which the length is
      shrunken or expanded.

    May return `None` if no sketch parameters were set.

### Function: set_sketch_params(self, scale, length, randomness)

**Description:** Set the sketch parameters.

Parameters
----------
scale : float, optional
    The amplitude of the wiggle perpendicular to the source line, in
    pixels.  If scale is `None`, or not provided, no sketch filter will
    be provided.
length : float, default: 128
    The length of the wiggle along the line, in pixels.
randomness : float, default: 16
    The scale factor by which the length is shrunken or expanded.

### Function: __init__(self, interval, callbacks)

**Description:** Parameters
----------
interval : int, default: 1000ms
    The time between timer events in milliseconds.  Will be stored as
    ``timer.interval``.
callbacks : list[tuple[callable, tuple, dict]]
    List of (func, args, kwargs) tuples that will be called upon timer
    events.  This list is accessible as ``timer.callbacks`` and can be
    manipulated directly, or the functions `~.TimerBase.add_callback`
    and `~.TimerBase.remove_callback` can be used.

### Function: __del__(self)

**Description:** Need to stop timer and possibly disconnect timer.

### Function: start(self, interval)

**Description:** Start the timer object.

Parameters
----------
interval : int, optional
    Timer interval in milliseconds; overrides a previously set interval
    if provided.

### Function: stop(self)

**Description:** Stop the timer.

### Function: _timer_start(self)

### Function: _timer_stop(self)

### Function: interval(self)

**Description:** The time between timer events, in milliseconds.

### Function: interval(self, interval)

### Function: single_shot(self)

**Description:** Whether this timer should stop after a single run.

### Function: single_shot(self, ss)

### Function: add_callback(self, func)

**Description:** Register *func* to be called by timer when the event fires. Any
additional arguments provided will be passed to *func*.

This function returns *func*, which makes it possible to use it as a
decorator.

### Function: remove_callback(self, func)

**Description:** Remove *func* from list of callbacks.

*args* and *kwargs* are optional and used to distinguish between copies
of the same function registered to be called with different arguments.
This behavior is deprecated.  In the future, ``*args, **kwargs`` won't
be considered anymore; to keep a specific callback removable by itself,
pass it to `add_callback` as a `functools.partial` object.

### Function: _timer_set_interval(self)

**Description:** Used to set interval on underlying timer object.

### Function: _timer_set_single_shot(self)

**Description:** Used to set single shot on underlying timer object.

### Function: _on_timer(self)

**Description:** Runs all function that have been registered as callbacks. Functions
can return False (or 0) if they should not be called any more. If there
are no callbacks, the timer is automatically stopped.

### Function: __init__(self, name, canvas, guiEvent)

### Function: _process(self)

**Description:** Process this event on ``self.canvas``, then unset ``guiEvent``.

### Function: __init__(self, name, canvas, renderer)

### Function: __init__(self, name, canvas)

### Function: __init__(self, name, canvas, x, y, guiEvent)

### Function: _set_inaxes(self, inaxes, xy)

### Function: __init__(self, name, canvas, x, y, button, key, step, dblclick, guiEvent)

### Function: __str__(self)

### Function: __init__(self, name, canvas, mouseevent, artist, guiEvent)

### Function: __init__(self, name, canvas, key, x, y, guiEvent)

## Class: Done

### Function: _draw(renderer)

### Function: save_args_and_handle_sigint()

### Function: supports_blit(cls)

**Description:** If this Canvas sub-class supports blitting.

### Function: __init__(self, figure)

### Function: _fix_ipython_backend2gui(cls)

### Function: new_manager(cls, figure, num)

**Description:** Create a new figure manager for *figure*, using this canvas class.

Notes
-----
This method should not be reimplemented in subclasses.  If
custom manager creation logic is needed, please reimplement
``FigureManager.create_with_canvas``.

### Function: _idle_draw_cntx(self)

### Function: is_saving(self)

**Description:** Return whether the renderer is in the process of saving
to a file, rather than rendering for an on-screen buffer.

### Function: blit(self, bbox)

**Description:** Blit the canvas in bbox (default entire canvas).

### Function: inaxes(self, xy)

**Description:** Return the topmost visible `~.axes.Axes` containing the point *xy*.

Parameters
----------
xy : (float, float)
    (x, y) pixel positions from left/bottom of the canvas.

Returns
-------
`~matplotlib.axes.Axes` or None
    The topmost visible Axes containing the point, or None if there
    is no Axes at the point.

### Function: grab_mouse(self, ax)

**Description:** Set the child `~.axes.Axes` which is grabbing the mouse events.

Usually called by the widgets themselves. It is an error to call this
if the mouse is already grabbed by another Axes.

### Function: release_mouse(self, ax)

**Description:** Release the mouse grab held by the `~.axes.Axes` *ax*.

Usually called by the widgets. It is ok to call this even if *ax*
doesn't have the mouse grab currently.

### Function: set_cursor(self, cursor)

**Description:** Set the current cursor.

This may have no effect if the backend does not display anything.

If required by the backend, this method should trigger an update in
the backend event loop after the cursor is set, as this method may be
called e.g. before a long-running task during which the GUI is not
updated.

Parameters
----------
cursor : `.Cursors`
    The cursor to display over the canvas. Note: some backends may
    change the cursor for the entire window.

### Function: draw(self)

**Description:** Render the `.Figure`.

This method must walk the artist tree, even if no output is produced,
because it triggers deferred work that users may want to access
before saving output to disk. For example computing limits,
auto-limits, and tick values.

### Function: draw_idle(self)

**Description:** Request a widget redraw once control returns to the GUI event loop.

Even if multiple calls to `draw_idle` occur before control returns
to the GUI event loop, the figure will only be rendered once.

Notes
-----
Backends may choose to override the method and implement their own
strategy to prevent multiple renderings.

### Function: device_pixel_ratio(self)

**Description:** The ratio of physical to logical pixels used for the canvas on screen.

By default, this is 1, meaning physical and logical pixels are the same
size. Subclasses that support High DPI screens may set this property to
indicate that said ratio is different. All Matplotlib interaction,
unless working directly with the canvas, remains in logical pixels.

### Function: _set_device_pixel_ratio(self, ratio)

**Description:** Set the ratio of physical to logical pixels used for the canvas.

Subclasses that support High DPI screens can set this property to
indicate that said ratio is different. The canvas itself will be
created at the physical size, while the client side will use the
logical size. Thus the DPI of the Figure will change to be scaled by
this ratio. Implementations that support High DPI screens should use
physical pixels for events so that transforms back to Axes space are
correct.

By default, this is 1, meaning physical and logical pixels are the same
size.

Parameters
----------
ratio : float
    The ratio of logical to physical pixels used for the canvas.

Returns
-------
bool
    Whether the ratio has changed. Backends may interpret this as a
    signal to resize the window, repaint the canvas, or change any
    other relevant properties.

### Function: get_width_height(self)

**Description:** Return the figure width and height in integral points or pixels.

When the figure is used on High DPI screens (and the backend supports
it), the truncation to integers occurs after scaling by the device
pixel ratio.

Parameters
----------
physical : bool, default: False
    Whether to return true physical pixels or logical pixels. Physical
    pixels may be used by backends that support HiDPI, but still
    configure the canvas using its actual size.

Returns
-------
width, height : int
    The size of the figure, in points or pixels, depending on the
    backend.

### Function: get_supported_filetypes(cls)

**Description:** Return dict of savefig file formats supported by this backend.

### Function: get_supported_filetypes_grouped(cls)

**Description:** Return a dict of savefig file formats supported by this backend,
where the keys are a file type name, such as 'Joint Photographic
Experts Group', and the values are a list of filename extensions used
for that filetype, such as ['jpg', 'jpeg'].

### Function: _switch_canvas_and_return_print_method(self, fmt, backend)

**Description:** Context manager temporarily setting the canvas for saving the figure::

    with (canvas._switch_canvas_and_return_print_method(fmt, backend)
          as print_method):
        # ``print_method`` is a suitable ``print_{fmt}`` method, and
        # the figure's canvas is temporarily switched to the method's
        # canvas within the with... block.  ``print_method`` is also
        # wrapped to suppress extra kwargs passed by ``print_figure``.

Parameters
----------
fmt : str
    If *backend* is None, then determine a suitable canvas class for
    saving to format *fmt* -- either the current canvas class, if it
    supports *fmt*, or whatever `get_registered_canvas_class` returns;
    switch the figure canvas to that canvas class.
backend : str or None, default: None
    If not None, switch the figure canvas to the ``FigureCanvas`` class
    of the given backend.

### Function: print_figure(self, filename, dpi, facecolor, edgecolor, orientation, format)

**Description:** Render the figure to hardcopy. Set the figure patch face and edge
colors.  This is useful because some of the GUIs have a gray figure
face color background and you'll probably want to override this on
hardcopy.

Parameters
----------
filename : str or path-like or file-like
    The file where the figure is saved.

dpi : float, default: :rc:`savefig.dpi`
    The dots per inch to save the figure in.

facecolor : :mpltype:`color` or 'auto', default: :rc:`savefig.facecolor`
    The facecolor of the figure.  If 'auto', use the current figure
    facecolor.

edgecolor : :mpltype:`color` or 'auto', default: :rc:`savefig.edgecolor`
    The edgecolor of the figure.  If 'auto', use the current figure
    edgecolor.

orientation : {'landscape', 'portrait'}, default: 'portrait'
    Only currently applies to PostScript printing.

format : str, optional
    Force a specific file format. If not given, the format is inferred
    from the *filename* extension, and if that fails from
    :rc:`savefig.format`.

bbox_inches : 'tight' or `.Bbox`, default: :rc:`savefig.bbox`
    Bounding box in inches: only the given portion of the figure is
    saved.  If 'tight', try to figure out the tight bbox of the figure.

pad_inches : float or 'layout', default: :rc:`savefig.pad_inches`
    Amount of padding in inches around the figure when bbox_inches is
    'tight'. If 'layout' use the padding from the constrained or
    compressed layout engine; ignored if one of those engines is not in
    use.

bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional
    A list of extra artists that will be considered when the
    tight bbox is calculated.

backend : str, optional
    Use a non-default backend to render the file, e.g. to render a
    png file with the "cairo" backend rather than the default "agg",
    or a pdf file with the "pgf" backend rather than the default
    "pdf".  Note that the default backend is normally sufficient.  See
    :ref:`the-builtin-backends` for a list of valid backends for each
    file format.  Custom backends can be referenced as "module://...".

### Function: get_default_filetype(cls)

**Description:** Return the default savefig file format as specified in
:rc:`savefig.format`.

The returned string does not include a period. This method is
overridden in backends that only support a single file type.

### Function: get_default_filename(self)

**Description:** Return a suitable default filename, including the extension.

### Function: mpl_connect(self, s, func)

**Description:** Bind function *func* to event *s*.

Parameters
----------
s : str
    One of the following events ids:

    - 'button_press_event'
    - 'button_release_event'
    - 'draw_event'
    - 'key_press_event'
    - 'key_release_event'
    - 'motion_notify_event'
    - 'pick_event'
    - 'resize_event'
    - 'scroll_event'
    - 'figure_enter_event',
    - 'figure_leave_event',
    - 'axes_enter_event',
    - 'axes_leave_event'
    - 'close_event'.

func : callable
    The callback function to be executed, which must have the
    signature::

        def func(event: Event) -> Any

    For the location events (button and key press/release), if the
    mouse is over the Axes, the ``inaxes`` attribute of the event will
    be set to the `~matplotlib.axes.Axes` the event occurs is over, and
    additionally, the variables ``xdata`` and ``ydata`` attributes will
    be set to the mouse location in data coordinates.  See `.KeyEvent`
    and `.MouseEvent` for more info.

    .. note::

        If func is a method, this only stores a weak reference to the
        method. Thus, the figure does not influence the lifetime of
        the associated object. Usually, you want to make sure that the
        object is kept alive throughout the lifetime of the figure by
        holding a reference to it.

Returns
-------
cid
    A connection id that can be used with
    `.FigureCanvasBase.mpl_disconnect`.

Examples
--------
::

    def on_press(event):
        print('you pressed', event.button, event.xdata, event.ydata)

    cid = canvas.mpl_connect('button_press_event', on_press)

### Function: mpl_disconnect(self, cid)

**Description:** Disconnect the callback with id *cid*.

Examples
--------
::

    cid = canvas.mpl_connect('button_press_event', on_press)
    # ... later
    canvas.mpl_disconnect(cid)

### Function: new_timer(self, interval, callbacks)

**Description:** Create a new backend-specific subclass of `.Timer`.

This is useful for getting periodic events through the backend's native
event loop.  Implemented only for backends with GUIs.

Parameters
----------
interval : int
    Timer interval in milliseconds.

callbacks : list[tuple[callable, tuple, dict]]
    Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``
    will be executed by the timer every *interval*.

    Callbacks which return ``False`` or ``0`` will be removed from the
    timer.

Examples
--------
>>> timer = fig.canvas.new_timer(callbacks=[(f1, (1,), {'a': 3})])

### Function: flush_events(self)

**Description:** Flush the GUI events for the figure.

Interactive backends need to reimplement this method.

### Function: start_event_loop(self, timeout)

**Description:** Start a blocking event loop.

Such an event loop is used by interactive functions, such as
`~.Figure.ginput` and `~.Figure.waitforbuttonpress`, to wait for
events.

The event loop blocks until a callback function triggers
`stop_event_loop`, or *timeout* is reached.

If *timeout* is 0 or negative, never timeout.

Only interactive backends need to reimplement this method and it relies
on `flush_events` being properly implemented.

Interactive backends should implement this in a more native way.

### Function: stop_event_loop(self)

**Description:** Stop the current blocking event loop.

Interactive backends need to reimplement this to match
`start_event_loop`

### Function: _get_uniform_gridstate(ticks)

### Function: __init__(self, canvas, num)

### Function: create_with_canvas(cls, canvas_class, figure, num)

**Description:** Create a manager for a given *figure* using a specific *canvas_class*.

Backends should override this method if they have specific needs for
setting up the canvas or the manager.

### Function: start_main_loop(cls)

**Description:** Start the main event loop.

This method is called by `.FigureManagerBase.pyplot_show`, which is the
implementation of `.pyplot.show`.  To customize the behavior of
`.pyplot.show`, interactive backends should usually override
`~.FigureManagerBase.start_main_loop`; if more customized logic is
necessary, `~.FigureManagerBase.pyplot_show` can also be overridden.

### Function: pyplot_show(cls)

**Description:** Show all figures.  This method is the implementation of `.pyplot.show`.

To customize the behavior of `.pyplot.show`, interactive backends
should usually override `~.FigureManagerBase.start_main_loop`; if more
customized logic is necessary, `~.FigureManagerBase.pyplot_show` can
also be overridden.

Parameters
----------
block : bool, optional
    Whether to block by calling ``start_main_loop``.  The default,
    None, means to block if we are neither in IPython's ``%pylab`` mode
    nor in ``interactive`` mode.

### Function: show(self)

**Description:** For GUI backends, show the figure window and redraw.
For non-GUI backends, raise an exception, unless running headless (i.e.
on Linux with an unset DISPLAY); this exception is converted to a
warning in `.Figure.show`.

### Function: destroy(self)

### Function: full_screen_toggle(self)

### Function: resize(self, w, h)

**Description:** For GUI backends, resize the window (in physical pixels).

### Function: get_window_title(self)

**Description:** Return the title text of the window containing the figure.

### Function: set_window_title(self, title)

**Description:** Set the title text of the window containing the figure.

Examples
--------
>>> fig = plt.figure()
>>> fig.canvas.manager.set_window_title('My figure')

### Function: __str__(self)

### Function: _navigate_mode(self)

### Function: __init__(self, canvas)

### Function: set_message(self, s)

**Description:** Display a message on toolbar or in status bar.

### Function: draw_rubberband(self, event, x0, y0, x1, y1)

**Description:** Draw a rectangle rubberband to indicate zoom limits.

Note that it is not guaranteed that ``x0 <= x1`` and ``y0 <= y1``.

### Function: remove_rubberband(self)

**Description:** Remove the rubberband.

### Function: home(self)

**Description:** Restore the original view.

For convenience of being directly connected as a GUI callback, which
often get passed additional parameters, this method accepts arbitrary
parameters, but does not use them.

### Function: back(self)

**Description:** Move back up the view lim stack.

For convenience of being directly connected as a GUI callback, which
often get passed additional parameters, this method accepts arbitrary
parameters, but does not use them.

### Function: forward(self)

**Description:** Move forward in the view lim stack.

For convenience of being directly connected as a GUI callback, which
often get passed additional parameters, this method accepts arbitrary
parameters, but does not use them.

### Function: _update_cursor(self, event)

**Description:** Update the cursor after a mouse move event or a tool (de)activation.

### Function: _wait_cursor_for_draw_cm(self)

**Description:** Set the cursor to a wait cursor when drawing the canvas.

In order to avoid constantly changing the cursor when the canvas
changes frequently, do nothing if this context was triggered during the
last second.  (Optimally we'd prefer only setting the wait cursor if
the *current* draw takes too long, but the current draw blocks the GUI
thread).

### Function: _mouse_event_to_message(event)

### Function: mouse_move(self, event)

### Function: _zoom_pan_handler(self, event)

### Function: _start_event_axes_interaction(self, event)

### Function: pan(self)

**Description:** Toggle the pan/zoom tool.

Pan with left button, zoom with right.

### Function: press_pan(self, event)

**Description:** Callback for mouse button press in pan/zoom mode.

### Function: drag_pan(self, event)

**Description:** Callback for dragging in pan/zoom mode.

### Function: release_pan(self, event)

**Description:** Callback for mouse button release in pan/zoom mode.

### Function: zoom(self)

### Function: press_zoom(self, event)

**Description:** Callback for mouse button press in zoom to rect mode.

### Function: drag_zoom(self, event)

**Description:** Callback for dragging in zoom mode.

### Function: release_zoom(self, event)

**Description:** Callback for mouse button release in zoom to rect mode.

### Function: push_current(self)

**Description:** Push the current view limits and position onto the stack.

### Function: _update_view(self)

**Description:** Update the viewlim and position from the view and position stack for
each Axes.

### Function: configure_subplots(self)

### Function: save_figure(self)

**Description:** Save the current figure.

Backend implementations may choose to return
the absolute path of the saved file, if any, as
a string.

If no file is created then `None` is returned.

If the backend does not implement this functionality
then `NavigationToolbar2.UNKNOWN_SAVED_STATUS` is returned.

Returns
-------
str or `NavigationToolbar2.UNKNOWN_SAVED_STATUS` or `None`
    The filepath of the saved figure.
    Returns `None` if figure is not saved.
    Returns `NavigationToolbar2.UNKNOWN_SAVED_STATUS` when
    the backend does not provide the information.

### Function: update(self)

**Description:** Reset the Axes stack.

### Function: set_history_buttons(self)

**Description:** Enable or disable the back/forward button.

### Function: __init__(self, toolmanager)

### Function: _tool_toggled_cbk(self, event)

**Description:** Capture the 'tool_trigger_[name]'

This only gets used for toggled tools.

### Function: add_tool(self, tool, group, position)

**Description:** Add a tool to this container.

Parameters
----------
tool : tool_like
    The tool to add, see `.ToolManager.get_tool`.
group : str
    The name of the group to add this tool to.
position : int, default: -1
    The position within the group to place this tool.

### Function: _get_image_filename(self, tool)

**Description:** Resolve a tool icon's filename.

### Function: trigger_tool(self, name)

**Description:** Trigger the tool.

Parameters
----------
name : str
    Name (id) of the tool triggered from within the container.

### Function: add_toolitem(self, name, group, position, image, description, toggle)

**Description:** A hook to add a toolitem to the container.

This hook must be implemented in each backend and contains the
backend-specific code to add an element to the toolbar.

.. warning::
    This is part of the backend implementation and should
    not be called by end-users.  They should instead call
    `.ToolContainerBase.add_tool`.

The callback associated with the button click event
must be *exactly* ``self.trigger_tool(name)``.

Parameters
----------
name : str
    Name of the tool to add, this gets used as the tool's ID and as the
    default label of the buttons.
group : str
    Name of the group that this tool belongs to.
position : int
    Position of the tool within its group, if -1 it goes at the end.
image : str
    Filename of the image for the button or `None`.
description : str
    Description of the tool, used for the tooltips.
toggle : bool
    * `True` : The button is a toggle (change the pressed/unpressed
      state between consecutive clicks).
    * `False` : The button is a normal button (returns to unpressed
      state after release).

### Function: toggle_toolitem(self, name, toggled)

**Description:** A hook to toggle a toolitem without firing an event.

This hook must be implemented in each backend and contains the
backend-specific code to silently toggle a toolbar element.

.. warning::
    This is part of the backend implementation and should
    not be called by end-users.  They should instead call
    `.ToolManager.trigger_tool` or `.ToolContainerBase.trigger_tool`
    (which are equivalent).

Parameters
----------
name : str
    Id of the tool to toggle.
toggled : bool
    Whether to set this tool as toggled or not.

### Function: remove_toolitem(self, name)

**Description:** A hook to remove a toolitem from the container.

This hook must be implemented in each backend and contains the
backend-specific code to remove an element from the toolbar; it is
called when `.ToolManager` emits a ``tool_removed_event``.

Because some tools are present only on the `.ToolManager` but not on
the `ToolContainer`, this method must be a no-op when called on a tool
absent from the container.

.. warning::
    This is part of the backend implementation and should
    not be called by end-users.  They should instead call
    `.ToolManager.remove_tool`.

Parameters
----------
name : str
    Name of the tool to remove.

### Function: set_message(self, s)

**Description:** Display a message on the toolbar.

Parameters
----------
s : str
    Message text.

### Function: new_figure_manager(cls, num)

**Description:** Create a new figure manager instance.

### Function: new_figure_manager_given_figure(cls, num, figure)

**Description:** Create a new figure manager instance for the given figure.

### Function: draw_if_interactive(cls)

### Function: show(cls)

**Description:** Show all figures.

`show` blocks by calling `mainloop` if *block* is ``True``, or if it is
``None`` and we are not in `interactive` mode and if IPython's
``%matplotlib`` integration has not been activated.

### Function: export(cls)

### Function: __call__(self, block)

### Function: cycle_or_default(seq, default)

### Function: notify_axes_change(fig)

### Function: _ax_filter(ax)

### Function: _capture_events(ax)

### Function: on_tool_fig_close(e)

## Class: Show

### Function: mainloop(self)
