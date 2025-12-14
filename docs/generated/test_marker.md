## AI Summary

A file named test_marker.py.


### Function: test_marker_fillstyle()

### Function: test_markers_valid(marker)

### Function: test_markers_invalid(marker)

## Class: UnsnappedMarkerStyle

**Description:** A MarkerStyle where the snap threshold is force-disabled.

This is used to compare to polygon/star/asterisk markers which do not have
any snap threshold set.

### Function: test_poly_marker(fig_test, fig_ref)

### Function: test_star_marker()

### Function: test_asterisk_marker(fig_test, fig_ref, request)

### Function: test_text_marker(fig_ref, fig_test)

### Function: test_marker_clipping(fig_ref, fig_test)

### Function: test_marker_init_transforms()

**Description:** Test that initializing marker with transform is a simple addition.

### Function: test_marker_init_joinstyle()

### Function: test_marker_init_captyle()

### Function: test_marker_transformed(marker, transform, expected)

### Function: test_marker_rotated_invalid()

### Function: test_marker_rotated(marker, deg, rad, expected)

### Function: test_marker_scaled()

### Function: test_alt_transform()

### Function: _recache(self)

### Function: draw_ref_marker(y, style, size)
