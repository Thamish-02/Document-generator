## AI Summary

A file named backend_gtk3.py.


### Function: _mpl_to_gtk_cursor(mpl_cursor)

## Class: FigureCanvasGTK3

## Class: NavigationToolbar2GTK3

## Class: ToolbarGTK3

## Class: SaveFigureGTK3

## Class: HelpGTK3

## Class: ToolCopyToClipboardGTK3

## Class: FigureManagerGTK3

## Class: _BackendGTK3

### Function: __init__(self, figure)

### Function: destroy(self)

### Function: set_cursor(self, cursor)

### Function: _mpl_coords(self, event)

**Description:** Convert the position of a GTK event, or of the current cursor position
if *event* is None, to Matplotlib coordinates.

GTK use logical pixels, but the figure is scaled to physical pixels for
rendering.  Transform to physical pixels so that all of the down-stream
transforms work as expected.

Also, the origin is different and needs to be corrected.

### Function: scroll_event(self, widget, event)

### Function: button_press_event(self, widget, event)

### Function: button_release_event(self, widget, event)

### Function: key_press_event(self, widget, event)

### Function: key_release_event(self, widget, event)

### Function: motion_notify_event(self, widget, event)

### Function: enter_notify_event(self, widget, event)

### Function: leave_notify_event(self, widget, event)

### Function: size_allocate(self, widget, allocation)

### Function: _mpl_buttons(event_state)

### Function: _mpl_modifiers(event_state)

### Function: _get_key(self, event)

### Function: _update_device_pixel_ratio(self)

### Function: configure_event(self, widget, event)

### Function: _draw_rubberband(self, rect)

### Function: _post_draw(self, widget, ctx)

### Function: on_draw_event(self, widget, ctx)

### Function: draw(self)

### Function: draw_idle(self)

### Function: flush_events(self)

### Function: __init__(self, canvas)

### Function: save_figure(self)

### Function: __init__(self, toolmanager)

### Function: add_toolitem(self, name, group, position, image_file, description, toggle)

### Function: _add_button(self, button, group, position)

### Function: _call_tool(self, btn, name)

### Function: toggle_toolitem(self, name, toggled)

### Function: remove_toolitem(self, name)

### Function: _add_separator(self)

### Function: set_message(self, s)

### Function: trigger(self)

### Function: _normalize_shortcut(self, key)

**Description:** Convert Matplotlib key presses to GTK+ accelerator identifiers.

Related to `FigureCanvasGTK3._get_key`.

### Function: _is_valid_shortcut(self, key)

**Description:** Check for a valid shortcut to be displayed.

- GTK will never send 'cmd+' (see `FigureCanvasGTK3._get_key`).
- The shortcut window only shows keyboard shortcuts, not mouse buttons.

### Function: _show_shortcuts_window(self)

### Function: _show_shortcuts_dialog(self)

### Function: trigger(self)

### Function: trigger(self)

### Function: idle_draw()

### Function: on_notify_filter()
