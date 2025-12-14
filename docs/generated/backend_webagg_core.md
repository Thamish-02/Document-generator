## AI Summary

A file named backend_webagg_core.py.


### Function: _handle_key(key)

**Description:** Handle key values

## Class: TimerTornado

## Class: TimerAsyncio

## Class: FigureCanvasWebAggCore

## Class: NavigationToolbar2WebAgg

## Class: FigureManagerWebAgg

## Class: _BackendWebAggCoreAgg

### Function: __init__(self)

### Function: _timer_start(self)

### Function: _timer_stop(self)

### Function: _timer_set_interval(self)

### Function: __init__(self)

### Function: _timer_start(self)

### Function: _timer_stop(self)

### Function: _timer_set_interval(self)

### Function: __init__(self)

### Function: show(self)

### Function: draw(self)

### Function: blit(self, bbox)

### Function: draw_idle(self)

### Function: set_cursor(self, cursor)

### Function: set_image_mode(self, mode)

**Description:** Set the image mode for any subsequent images which will be sent
to the clients. The modes may currently be either 'full' or 'diff'.

Note: diff images may not contain transparency, therefore upon
draw this mode may be changed if the resulting image has any
transparent component.

### Function: get_diff_image(self)

### Function: handle_event(self, event)

### Function: handle_unknown_event(self, event)

### Function: handle_ack(self, event)

### Function: handle_draw(self, event)

### Function: _handle_mouse(self, event)

### Function: _handle_key(self, event)

### Function: handle_toolbar_button(self, event)

### Function: handle_refresh(self, event)

### Function: handle_resize(self, event)

### Function: handle_send_image_mode(self, event)

### Function: handle_set_device_pixel_ratio(self, event)

### Function: handle_set_dpi_ratio(self, event)

### Function: _handle_set_device_pixel_ratio(self, device_pixel_ratio)

### Function: send_event(self, event_type)

### Function: __init__(self, canvas)

### Function: set_message(self, message)

### Function: draw_rubberband(self, event, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: save_figure(self)

**Description:** Save the current figure.

### Function: pan(self)

### Function: zoom(self)

### Function: set_history_buttons(self)

### Function: __init__(self, canvas, num)

### Function: show(self)

### Function: resize(self, w, h, forward)

### Function: set_window_title(self, title)

### Function: get_window_title(self)

### Function: add_web_socket(self, web_socket)

### Function: remove_web_socket(self, web_socket)

### Function: handle_json(self, content)

### Function: refresh_all(self)

### Function: get_javascript(cls, stream)

### Function: get_static_file_path(cls)

### Function: _send_event(self, event_type)
