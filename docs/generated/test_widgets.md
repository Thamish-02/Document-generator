## AI Summary

A file named test_widgets.py.


### Function: ax()

### Function: test_save_blitted_widget_as_pdf()

### Function: test_rectangle_selector(ax, kwargs)

### Function: test_rectangle_minspan(ax, spancoords, minspanx, x1, minspany, y1)

### Function: test_rectangle_drag(ax, drag_from_anywhere, new_center)

### Function: test_rectangle_selector_set_props_handle_props(ax)

### Function: test_rectangle_resize(ax)

### Function: test_rectangle_add_state(ax)

### Function: test_rectangle_resize_center(ax, add_state)

### Function: test_rectangle_resize_square(ax, add_state)

### Function: test_rectangle_resize_square_center(ax)

### Function: test_rectangle_rotate(ax, selector_class)

### Function: test_rectangle_add_remove_set(ax)

### Function: test_rectangle_resize_square_center_aspect(ax, use_data_coordinates)

### Function: test_ellipse(ax)

**Description:** For ellipse, test out the key modifiers

### Function: test_rectangle_handles(ax)

### Function: test_rectangle_selector_onselect(ax, interactive)

### Function: test_rectangle_selector_ignore_outside(ax, ignore_event_outside)

### Function: test_span_selector(ax, orientation, onmove_callback, kwargs)

### Function: test_span_selector_onselect(ax, interactive)

### Function: test_span_selector_ignore_outside(ax, ignore_event_outside)

### Function: test_span_selector_drag(ax, drag_from_anywhere)

### Function: test_span_selector_direction(ax)

### Function: test_span_selector_set_props_handle_props(ax)

### Function: test_selector_clear(ax, selector)

### Function: test_selector_clear_method(ax, selector)

### Function: test_span_selector_add_state(ax)

### Function: test_tool_line_handle(ax)

### Function: test_span_selector_bound(direction)

### Function: test_span_selector_animated_artists_callback()

**Description:** Check that the animated artists changed in callbacks are updated.

### Function: test_snapping_values_span_selector(ax)

### Function: test_span_selector_snap(ax)

### Function: test_span_selector_extents(ax)

### Function: test_lasso_selector(ax, kwargs)

### Function: test_lasso_selector_set_props(ax)

### Function: test_lasso_set_props(ax)

### Function: test_CheckButtons(ax)

### Function: test_TextBox(ax, toolbar)

### Function: test_RadioButtons(ax)

### Function: test_check_radio_buttons_image()

### Function: test_radio_buttons(fig_test, fig_ref)

### Function: test_radio_buttons_props(fig_test, fig_ref)

### Function: test_radio_button_active_conflict(ax)

### Function: test_radio_buttons_activecolor_change(fig_test, fig_ref)

### Function: test_check_buttons(fig_test, fig_ref)

### Function: test_check_button_props(fig_test, fig_ref)

### Function: test_slider_slidermin_slidermax_invalid()

### Function: test_slider_slidermin_slidermax()

### Function: test_slider_valmin_valmax()

### Function: test_slider_valstep_snapping()

### Function: test_slider_horizontal_vertical()

### Function: test_slider_reset()

### Function: test_range_slider(orientation)

### Function: test_range_slider_same_init_values(orientation)

### Function: check_polygon_selector(event_sequence, expected_result, selections_count)

**Description:** Helper function to test Polygon Selector.

Parameters
----------
event_sequence : list of tuples (etype, dict())
    A sequence of events to perform. The sequence is a list of tuples
    where the first element of the tuple is an etype (e.g., 'onmove',
    'press', etc.), and the second element of the tuple is a dictionary of
     the arguments for the event (e.g., xdata=5, key='shift', etc.).
expected_result : list of vertices (xdata, ydata)
    The list of vertices that are expected to result from the event
    sequence.
selections_count : int
    Wait for the tool to call its `onselect` function `selections_count`
    times, before comparing the result to the `expected_result`
**kwargs
    Keyword arguments are passed to PolygonSelector.

### Function: polygon_place_vertex(xdata, ydata)

### Function: polygon_remove_vertex(xdata, ydata)

### Function: test_polygon_selector(draw_bounding_box)

### Function: test_polygon_selector_set_props_handle_props(ax, draw_bounding_box)

### Function: test_rect_visibility(fig_test, fig_ref)

### Function: test_polygon_selector_remove(idx, draw_bounding_box)

### Function: test_polygon_selector_remove_first_point(draw_bounding_box)

### Function: test_polygon_selector_redraw(ax, draw_bounding_box)

### Function: test_polygon_selector_verts_setter(fig_test, fig_ref, draw_bounding_box)

### Function: test_polygon_selector_box(ax)

### Function: test_polygon_selector_clear_method(ax)

### Function: test_MultiCursor(horizOn, vertOn)

### Function: mean(vmin, vmax)

### Function: onselect()

### Function: onselect(vmin, vmax)

### Function: handle_positions(slider)
