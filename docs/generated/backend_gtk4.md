## AI Summary

A file named backend_gtk4.py.


## Class: FigureCanvasGTK4

## Class: NavigationToolbar2GTK4

## Class: ToolbarGTK4

## Class: SaveFigureGTK4

## Class: HelpGTK4

## Class: ToolCopyToClipboardGTK4

## Class: FigureManagerGTK4

## Class: _BackendGTK4

### Function: __init__(self, figure)

### Function: destroy(self)

### Function: set_cursor(self, cursor)

### Function: _mpl_coords(self, xy)

**Description:** Convert the *xy* position of a GTK event, or of the current cursor
position if *xy* is None, to Matplotlib coordinates.

GTK use logical pixels, but the figure is scaled to physical pixels for
rendering.  Transform to physical pixels so that all of the down-stream
transforms work as expected.

Also, the origin is different and needs to be corrected.

### Function: scroll_event(self, controller, dx, dy)

### Function: button_press_event(self, controller, n_press, x, y)

### Function: button_release_event(self, controller, n_press, x, y)

### Function: key_press_event(self, controller, keyval, keycode, state)

### Function: key_release_event(self, controller, keyval, keycode, state)

### Function: motion_notify_event(self, controller, x, y)

### Function: enter_notify_event(self, controller, x, y)

### Function: leave_notify_event(self, controller)

### Function: resize_event(self, area, width, height)

### Function: _mpl_buttons(self, controller)

### Function: _mpl_modifiers(self, controller)

### Function: _get_key(self, keyval, keycode, state)

### Function: _realize_event(self, obj)

### Function: _update_device_pixel_ratio(self)

### Function: _draw_rubberband(self, rect)

### Function: _draw_func(self, drawing_area, ctx, width, height)

### Function: _post_draw(self, widget, ctx)

### Function: on_draw_event(self, widget, ctx)

### Function: draw(self)

### Function: draw_idle(self)

### Function: flush_events(self)

### Function: __init__(self, canvas)

### Function: save_figure(self)

### Function: __init__(self, toolmanager)

### Function: add_toolitem(self, name, group, position, image_file, description, toggle)

### Function: _find_child_at_position(self, group, position)

### Function: _add_button(self, button, group, position)

### Function: _call_tool(self, btn, name)

### Function: toggle_toolitem(self, name, toggled)

### Function: remove_toolitem(self, name)

### Function: _add_separator(self)

### Function: set_message(self, s)

### Function: trigger(self)

### Function: _normalize_shortcut(self, key)

**Description:** Convert Matplotlib key presses to GTK+ accelerator identifiers.

Related to `FigureCanvasGTK4._get_key`.

### Function: _is_valid_shortcut(self, key)

**Description:** Check for a valid shortcut to be displayed.

- GTK will never send 'cmd+' (see `FigureCanvasGTK4._get_key`).
- The shortcut window only shows keyboard shortcuts, not mouse buttons.

### Function: trigger(self)

### Function: trigger(self)

### Function: idle_draw()

### Function: on_response(dialog, response)
