## AI Summary

A file named test_backend_qt.py.


### Function: qt_core(request)

### Function: test_fig_close()

### Function: test_correct_key(backend, qt_core, qt_key, qt_mods, answer, monkeypatch)

**Description:** Make a figure.
Send a key_press_event event (using non-public, qtX backend specific api).
Catch the event.
Assert sent and caught keys are the same.

### Function: test_device_pixel_ratio_change(qt_core)

**Description:** Make sure that if the pixel ratio changes, the figure dpi changes but the
widget remains the same logical size.

### Function: test_subplottool()

### Function: test_figureoptions()

### Function: test_save_figure_return(tmp_path)

### Function: test_figureoptions_with_datetime_axes()

### Function: test_double_resize()

### Function: test_canvas_reinit()

### Function: test_form_widget_get_with_datetime_and_date_fields()

### Function: _get_testable_qt_backends()

### Function: test_fig_sigint_override(qt_core)

### Function: test_ipython()

## Class: _Event

### Function: on_key_press(event)

### Function: crashing_callback(fig, stale)

### Function: fire_signal_and_quit()

### Function: custom_handler(signum, frame)

### Function: isAutoRepeat(self)

### Function: key(self)

### Function: set_device_pixel_ratio(ratio)
