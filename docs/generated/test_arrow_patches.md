## AI Summary

A file named test_arrow_patches.py.


### Function: draw_arrow(ax, t, r)

### Function: test_fancyarrow()

### Function: test_boxarrow()

### Function: __prepare_fancyarrow_dpi_cor_test()

**Description:** Convenience function that prepares and returns a FancyArrowPatch. It aims
at being used to test that the size of the arrow head does not depend on
the DPI value of the exported picture.

NB: this function *is not* a test in itself!

### Function: test_fancyarrow_dpi_cor_100dpi()

**Description:** Check the export of a FancyArrowPatch @ 100 DPI. FancyArrowPatch is
instantiated through a dedicated function because another similar test
checks a similar export but with a different DPI value.

Remark: test only a rasterized format.

### Function: test_fancyarrow_dpi_cor_200dpi()

**Description:** As test_fancyarrow_dpi_cor_100dpi, but exports @ 200 DPI. The relative size
of the arrow head should be the same.

### Function: test_fancyarrow_dash()

### Function: test_arrow_styles()

### Function: test_connection_styles()

### Function: test_invalid_intersection()
