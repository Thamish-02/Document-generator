## AI Summary

A file named _backend_tk.py.


### Function: _restore_foreground_window_at_end()

### Function: _blit(argsid)

**Description:** Thin wrapper to blit called via tkapp.call.

*argsid* is a unique string identifier to fetch the correct arguments from
the ``_blit_args`` dict, since arguments cannot be passed directly.

### Function: blit(photoimage, aggimage, offsets, bbox)

**Description:** Blit *aggimage* to *photoimage*.

*offsets* is a tuple describing how to fill the ``offset`` field of the
``Tk_PhotoImageBlock`` struct: it should be (0, 1, 2, 3) for RGBA8888 data,
(2, 1, 0, 3) for little-endian ARBG32 (i.e. GBRA8888) data and (1, 2, 3, 0)
for big-endian ARGB32 (i.e. ARGB8888) data.

If *bbox* is passed, it defines the region that gets blitted. That region
will be composed with the previous data according to the alpha channel.
Blitting will be clipped to pixels inside the canvas, including silently
doing nothing if the *bbox* region is entirely outside the canvas.

Tcl events must be dispatched to trigger a blit from a non-Tcl thread.

## Class: TimerTk

**Description:** Subclass of `backend_bases.TimerBase` using Tk timer events.

## Class: FigureCanvasTk

## Class: FigureManagerTk

**Description:** Attributes
----------
canvas : `FigureCanvas`
    The FigureCanvas instance
num : int or str
    The Figure number
toolbar : tk.Toolbar
    The tk.Toolbar
window : tk.Window
    The tk.Window

## Class: NavigationToolbar2Tk

### Function: add_tooltip(widget, text)

## Class: RubberbandTk

## Class: ToolbarTk

## Class: SaveFigureTk

## Class: ConfigureSubplotsTk

## Class: HelpTk

## Class: _BackendTk

### Function: __init__(self, parent)

### Function: _timer_start(self)

### Function: _timer_stop(self)

### Function: _on_timer(self)

### Function: __init__(self, figure, master)

### Function: _update_device_pixel_ratio(self, event)

### Function: resize(self, event)

### Function: draw_idle(self)

### Function: get_tk_widget(self)

**Description:** Return the Tk widget used to implement FigureCanvasTkAgg.

Although the initial implementation uses a Tk canvas,  this routine
is intended to hide that fact.

### Function: _event_mpl_coords(self, event)

### Function: motion_notify_event(self, event)

### Function: enter_notify_event(self, event)

### Function: leave_notify_event(self, event)

### Function: button_press_event(self, event, dblclick)

### Function: button_dblclick_event(self, event)

### Function: button_release_event(self, event)

### Function: scroll_event(self, event)

### Function: scroll_event_windows(self, event)

**Description:** MouseWheel event processor

### Function: _mpl_buttons(event)

### Function: _mpl_modifiers(event)

### Function: _get_key(self, event)

### Function: key_press(self, event)

### Function: key_release(self, event)

### Function: new_timer(self)

### Function: flush_events(self)

### Function: start_event_loop(self, timeout)

### Function: stop_event_loop(self)

### Function: set_cursor(self, cursor)

### Function: __init__(self, canvas, num, window)

### Function: create_with_canvas(cls, canvas_class, figure, num)

### Function: start_main_loop(cls)

### Function: _update_window_dpi(self)

### Function: resize(self, width, height)

### Function: show(self)

### Function: destroy(self)

### Function: get_window_title(self)

### Function: set_window_title(self, title)

### Function: full_screen_toggle(self)

### Function: __init__(self, canvas, window)

**Description:** Parameters
----------
canvas : `FigureCanvas`
    The figure canvas on which to operate.
window : tk.Window
    The tk.Window which owns this toolbar.
pack_toolbar : bool, default: True
    If True, add the toolbar to the parent's pack manager's packing
    list during initialization with ``side="bottom"`` and ``fill="x"``.
    If you want to use the toolbar with a different layout manager, use
    ``pack_toolbar=False``.

### Function: _rescale(self)

**Description:** Scale all children of the toolbar to current DPI setting.

Before this is called, the Tk scaling setting will have been updated to
match the new DPI. Tk widgets do not update for changes to scaling, but
all measurements made after the change will match the new scaling. Thus
this function re-applies all the same sizes in points, which Tk will
scale correctly to pixels.

### Function: _update_buttons_checked(self)

### Function: pan(self)

### Function: zoom(self)

### Function: set_message(self, s)

### Function: draw_rubberband(self, event, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: _set_image_for_button(self, button)

**Description:** Set the image for a button based on its pixel size.

The pixel size is determined by the DPI scaling of the window.

### Function: _Button(self, text, image_file, toggle, command)

### Function: _Spacer(self)

### Function: save_figure(self)

### Function: set_history_buttons(self)

### Function: showtip(event)

**Description:** Display text in tooltip window.

### Function: hidetip(event)

### Function: draw_rubberband(self, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: __init__(self, toolmanager, window)

### Function: _rescale(self)

### Function: add_toolitem(self, name, group, position, image_file, description, toggle)

### Function: _get_groupframe(self, group)

### Function: _add_separator(self)

### Function: _button_click(self, name)

### Function: toggle_toolitem(self, name, toggled)

### Function: remove_toolitem(self, name)

### Function: set_message(self, s)

### Function: trigger(self)

### Function: trigger(self)

### Function: trigger(self)

### Function: scroll_event_windows(event)

### Function: filter_destroy(event)

### Function: idle_draw()

### Function: delayed_destroy()

### Function: _get_color(color_name)

### Function: _is_dark(color)

### Function: _recolor_icon(image, color)

### Function: destroy()
