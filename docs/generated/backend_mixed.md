## AI Summary

A file named backend_mixed.py.


## Class: MixedModeRenderer

**Description:** A helper class to implement a renderer that switches between
vector and raster drawing.  An example may be a PDF writer, where
most things are drawn with PDF vector commands, but some very
complex objects, such as quad meshes, are rasterised and then
output as images.

### Function: __init__(self, figure, width, height, dpi, vector_renderer, raster_renderer_class, bbox_inches_restore)

**Description:** Parameters
----------
figure : `~matplotlib.figure.Figure`
    The figure instance.
width : float
    The width of the canvas in logical units
height : float
    The height of the canvas in logical units
dpi : float
    The dpi of the canvas
vector_renderer : `~matplotlib.backend_bases.RendererBase`
    An instance of a subclass of
    `~matplotlib.backend_bases.RendererBase` that will be used for the
    vector drawing.
raster_renderer_class : `~matplotlib.backend_bases.RendererBase`
    The renderer class to use for the raster drawing.  If not provided,
    this will use the Agg backend (which is currently the only viable
    option anyway.)

### Function: __getattr__(self, attr)

### Function: start_rasterizing(self)

**Description:** Enter "raster" mode.  All subsequent drawing commands (until
`stop_rasterizing` is called) will be drawn with the raster backend.

### Function: stop_rasterizing(self)

**Description:** Exit "raster" mode.  All of the drawing that was done since
the last `start_rasterizing` call will be copied to the
vector backend by calling draw_image.
