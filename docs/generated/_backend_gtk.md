## AI Summary

A file named _backend_gtk.py.


### Function: _shutdown_application(app)

### Function: _create_application()

### Function: mpl_to_gtk_cursor_name(mpl_cursor)

## Class: TimerGTK

**Description:** Subclass of `.TimerBase` using GTK timer events.

## Class: _FigureCanvasGTK

## Class: _FigureManagerGTK

**Description:** Attributes
----------
canvas : `FigureCanvas`
    The FigureCanvas instance
num : int or str
    The Figure number
toolbar : Gtk.Toolbar or Gtk.Box
    The toolbar
vbox : Gtk.VBox
    The Gtk.VBox containing the canvas and toolbar
window : Gtk.Window
    The Gtk.Window

## Class: _NavigationToolbar2GTK

## Class: RubberbandGTK

## Class: ConfigureSubplotsGTK

## Class: _BackendGTK

### Function: __init__(self)

### Function: _timer_start(self)

### Function: _timer_stop(self)

### Function: _timer_set_interval(self)

### Function: _on_timer(self)

### Function: __init__(self, canvas, num)

### Function: destroy(self)

### Function: start_main_loop(cls)

### Function: show(self)

### Function: full_screen_toggle(self)

### Function: get_window_title(self)

### Function: set_window_title(self, title)

### Function: resize(self, width, height)

### Function: set_message(self, s)

### Function: draw_rubberband(self, event, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: _update_buttons_checked(self)

### Function: pan(self)

### Function: zoom(self)

### Function: set_history_buttons(self)

### Function: draw_rubberband(self, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: trigger(self)
