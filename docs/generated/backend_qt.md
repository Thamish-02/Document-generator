## AI Summary

A file named backend_qt.py.


### Function: _create_qApp()

### Function: _allow_interrupt_qt(qapp_or_eventloop)

**Description:** A context manager that allows terminating a plot by sending a SIGINT.

## Class: TimerQT

**Description:** Subclass of `.TimerBase` using QTimer events.

## Class: FigureCanvasQT

## Class: MainWindow

## Class: FigureManagerQT

**Description:** Attributes
----------
canvas : `FigureCanvas`
    The FigureCanvas instance
num : int or str
    The Figure number
toolbar : qt.QToolBar
    The qt.QToolBar
window : qt.QMainWindow
    The qt.QMainWindow

## Class: NavigationToolbar2QT

## Class: SubplotToolQt

## Class: ToolbarQt

## Class: ConfigureSubplotsQt

## Class: SaveFigureQt

## Class: RubberbandQt

## Class: HelpQt

## Class: ToolCopyToClipboardQT

## Class: _BackendQT

### Function: prepare_notifier(rsock)

### Function: handle_sigint(sn)

### Function: __init__(self)

### Function: __del__(self)

### Function: _timer_set_single_shot(self)

### Function: _timer_set_interval(self)

### Function: _timer_start(self)

### Function: _timer_stop(self)

### Function: __init__(self, figure)

### Function: _update_pixel_ratio(self)

### Function: _update_screen(self, screen)

### Function: eventFilter(self, source, event)

### Function: showEvent(self, event)

### Function: set_cursor(self, cursor)

### Function: mouseEventCoords(self, pos)

**Description:** Calculate mouse coordinates in physical pixels.

Qt uses logical pixels, but the figure is scaled to physical
pixels for rendering.  Transform to physical pixels so that
all of the down-stream transforms work as expected.

Also, the origin is different and needs to be corrected.

### Function: enterEvent(self, event)

### Function: leaveEvent(self, event)

### Function: mousePressEvent(self, event)

### Function: mouseDoubleClickEvent(self, event)

### Function: mouseMoveEvent(self, event)

### Function: mouseReleaseEvent(self, event)

### Function: wheelEvent(self, event)

### Function: keyPressEvent(self, event)

### Function: keyReleaseEvent(self, event)

### Function: resizeEvent(self, event)

### Function: sizeHint(self)

### Function: minimumSizeHint(self)

### Function: _mpl_buttons(buttons)

### Function: _mpl_modifiers(modifiers)

### Function: _get_key(self, event)

### Function: flush_events(self)

### Function: start_event_loop(self, timeout)

### Function: stop_event_loop(self, event)

### Function: draw(self)

**Description:** Render the figure, and queue a request for a Qt draw.

### Function: draw_idle(self)

**Description:** Queue redraw of the Agg buffer and request Qt paintEvent.

### Function: blit(self, bbox)

### Function: _draw_idle(self)

### Function: drawRectangle(self, rect)

### Function: closeEvent(self, event)

### Function: __init__(self, canvas, num)

### Function: full_screen_toggle(self)

### Function: _widgetclosed(self)

### Function: resize(self, width, height)

### Function: start_main_loop(cls)

### Function: show(self)

### Function: destroy(self)

### Function: get_window_title(self)

### Function: set_window_title(self, title)

### Function: __init__(self, canvas, parent, coordinates)

**Description:** coordinates: should we show the coordinates on the right?

### Function: _icon(self, name)

**Description:** Construct a `.QIcon` from an image file *name*, including the extension
and relative to Matplotlib's "images" data directory.

### Function: edit_parameters(self)

### Function: _update_buttons_checked(self)

### Function: pan(self)

### Function: zoom(self)

### Function: set_message(self, s)

### Function: draw_rubberband(self, event, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: configure_subplots(self)

### Function: save_figure(self)

### Function: set_history_buttons(self)

### Function: __init__(self, targetfig, parent)

### Function: update_from_current_subplotpars(self)

### Function: _export_values(self)

### Function: _on_value_changed(self)

### Function: _tight_layout(self)

### Function: _reset(self)

### Function: __init__(self, toolmanager, parent)

### Function: add_toolitem(self, name, group, position, image_file, description, toggle)

### Function: _add_to_group(self, group, name, button, position)

### Function: toggle_toolitem(self, name, toggled)

### Function: remove_toolitem(self, name)

### Function: set_message(self, s)

### Function: __init__(self)

### Function: trigger(self)

### Function: trigger(self)

### Function: draw_rubberband(self, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: trigger(self)

### Function: trigger(self)

### Function: _may_clear_sock()

### Function: handler()

### Function: _draw_rect_callback(painter)

### Function: _draw_rect_callback(painter)
