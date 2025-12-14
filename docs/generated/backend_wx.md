## AI Summary

A file named backend_wx.py.


### Function: _create_wxapp()

## Class: TimerWx

**Description:** Subclass of `.TimerBase` using wx.Timer events.

## Class: RendererWx

**Description:** The renderer handles all the drawing primitives using a graphics
context instance that controls the colors/styles. It acts as the
'renderer' instance used by many classes in the hierarchy.

## Class: GraphicsContextWx

**Description:** The graphics context provides the color, line styles, etc.

This class stores a reference to a wxMemoryDC, and a
wxGraphicsContext that draws to it.  Creating a wxGraphicsContext
seems to be fairly heavy, so these objects are cached based on the
bitmap object that is passed in.

The base GraphicsContext stores colors as an RGB tuple on the unit
interval, e.g., (0.5, 0.0, 1.0).  wxPython uses an int interval, but
since wxPython colour management is rather simple, I have not chosen
to implement a separate colour manager class.

## Class: _FigureCanvasWxBase

**Description:** The FigureCanvas contains the figure and does event handling.

In the wxPython backend, it is derived from wxPanel, and (usually) lives
inside a frame instantiated by a FigureManagerWx. The parent window
probably implements a wx.Sizer to control the displayed control size - but
we give a hint as to our preferred minimum size.

## Class: FigureCanvasWx

## Class: FigureFrameWx

## Class: FigureManagerWx

**Description:** Container/controller for the FigureCanvas and GUI frame.

It is instantiated by Gcf whenever a new figure is created.  Gcf is
responsible for managing multiple instances of FigureManagerWx.

Attributes
----------
canvas : `FigureCanvas`
    a FigureCanvasWx(wx.Panel) instance
window : wxFrame
    a wxFrame instance - wxpython.org/Phoenix/docs/html/Frame.html

### Function: _load_bitmap(filename)

**Description:** Load a wx.Bitmap from a file in the "images" directory of the Matplotlib
data.

### Function: _set_frame_icon(frame)

## Class: NavigationToolbar2Wx

## Class: ToolbarWx

## Class: ConfigureSubplotsWx

## Class: SaveFigureWx

## Class: RubberbandWx

## Class: _HelpDialog

## Class: HelpWx

## Class: ToolCopyToClipboardWx

## Class: _BackendWx

### Function: __init__(self)

### Function: _timer_start(self)

### Function: _timer_stop(self)

### Function: _timer_set_interval(self)

### Function: __init__(self, bitmap, dpi)

**Description:** Initialise a wxWindows renderer instance.

### Function: flipy(self)

### Function: get_text_width_height_descent(self, s, prop, ismath)

### Function: get_canvas_width_height(self)

### Function: handle_clip_rectangle(self, gc)

### Function: convert_path(gfx_ctx, path, transform)

### Function: draw_path(self, gc, path, transform, rgbFace)

### Function: draw_image(self, gc, x, y, im)

### Function: draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)

### Function: new_gc(self)

### Function: get_wx_font(self, s, prop)

**Description:** Return a wx font.  Cache font instances for efficiency.

### Function: points_to_pixels(self, points)

### Function: __init__(self, bitmap, renderer)

### Function: select(self)

**Description:** Select the current bitmap into this wxDC instance.

### Function: unselect(self)

**Description:** Select a Null bitmap into this wxDC instance.

### Function: set_foreground(self, fg, isRGBA)

### Function: set_linewidth(self, w)

### Function: set_capstyle(self, cs)

### Function: set_joinstyle(self, js)

### Function: get_wxcolour(self, color)

**Description:** Convert an RGB(A) color to a wx.Colour.

### Function: __init__(self, parent, id, figure)

**Description:** Initialize a FigureWx instance.

- Initialize the FigureCanvasBase and wxPanel parents.
- Set event handlers for resize, paint, and keyboard and mouse
  interaction.

### Function: Copy_to_Clipboard(self, event)

**Description:** Copy bitmap of canvas to system clipboard.

### Function: _update_device_pixel_ratio(self)

### Function: draw_idle(self)

### Function: flush_events(self)

### Function: start_event_loop(self, timeout)

### Function: stop_event_loop(self, event)

### Function: _get_imagesave_wildcards(self)

**Description:** Return the wildcard string for the filesave dialog.

### Function: gui_repaint(self, drawDC)

**Description:** Update the displayed image on the GUI canvas, using the supplied
wx.PaintDC device context.

### Function: _on_paint(self, event)

**Description:** Called when wxPaintEvt is generated.

### Function: _on_size(self, event)

**Description:** Called when wxEventSize is generated.

In this application we attempt to resize to fit the window, so it
is better to take the performance hit and redraw the whole window.

### Function: _mpl_buttons()

### Function: _mpl_modifiers(event)

### Function: _get_key(self, event)

### Function: _mpl_coords(self, pos)

**Description:** Convert a wx position, defaulting to the current cursor position, to
Matplotlib coordinates.

### Function: _on_key_down(self, event)

**Description:** Capture key press.

### Function: _on_key_up(self, event)

**Description:** Release key.

### Function: set_cursor(self, cursor)

### Function: _set_capture(self, capture)

**Description:** Control wx mouse capture.

### Function: _on_capture_lost(self, event)

**Description:** Capture changed or lost

### Function: _on_mouse_button(self, event)

**Description:** Start measuring on an axis.

### Function: _on_mouse_wheel(self, event)

**Description:** Translate mouse wheel events into matplotlib events

### Function: _on_motion(self, event)

**Description:** Start measuring on an axis.

### Function: _on_enter(self, event)

**Description:** Mouse has entered the window.

### Function: _on_leave(self, event)

**Description:** Mouse has left the window.

### Function: draw(self, drawDC)

**Description:** Render the figure using RendererWx instance renderer, or using a
previously defined renderer if none is specified.

### Function: _print_image(self, filetype, filename)

### Function: __init__(self, num, fig)

### Function: _on_close(self, event)

### Function: __init__(self, canvas, num, frame)

### Function: create_with_canvas(cls, canvas_class, figure, num)

### Function: start_main_loop(cls)

### Function: show(self)

### Function: destroy(self)

### Function: full_screen_toggle(self)

### Function: get_window_title(self)

### Function: set_window_title(self, title)

### Function: resize(self, width, height)

### Function: __init__(self, canvas, coordinates)

### Function: _icon(name)

**Description:** Construct a `wx.Bitmap` suitable for use as icon from an image file
*name*, including the extension and relative to Matplotlib's "images"
data directory.

### Function: _update_buttons_checked(self)

### Function: zoom(self)

### Function: pan(self)

### Function: save_figure(self)

### Function: draw_rubberband(self, event, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: set_message(self, s)

### Function: set_history_buttons(self)

### Function: __init__(self, toolmanager, parent, style)

### Function: _get_tool_pos(self, tool)

**Description:** Find the position (index) of a wx.ToolBarToolBase in a ToolBar.

``ToolBar.GetToolPos`` is not useful because wx assigns the same Id to
all Separators and StretchableSpaces.

### Function: add_toolitem(self, name, group, position, image_file, description, toggle)

### Function: toggle_toolitem(self, name, toggled)

### Function: remove_toolitem(self, name)

### Function: set_message(self, s)

### Function: trigger(self)

### Function: trigger(self)

### Function: draw_rubberband(self, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: __init__(self, parent, help_entries)

### Function: _on_close(self, event)

### Function: show(cls, parent, help_entries)

### Function: trigger(self)

### Function: trigger(self)

### Function: handler(event)
