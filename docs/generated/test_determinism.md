## AI Summary

A file named test_determinism.py.


### Function: _save_figure(objects, fmt, usetex)

### Function: test_determinism_check(objects, fmt, usetex)

**Description:** Output three times the same graphs and checks that the outputs are exactly
the same.

Parameters
----------
objects : str
    Objects to be included in the test document: 'm' for markers, 'h' for
    hatch patterns, 'i' for images, and 'p' for paths.
fmt : {"pdf", "ps", "svg"}
    Output format.

### Function: test_determinism_source_date_epoch(fmt, string)

**Description:** Test SOURCE_DATE_EPOCH support. Output a document with the environment
variable SOURCE_DATE_EPOCH set to 2000-01-01 00:00 UTC and check that the
document contains the timestamp that corresponds to this date (given as an
argument).

Parameters
----------
fmt : {"pdf", "ps", "svg"}
    Output format.
string : bytes
    Timestamp string for 2000-01-01 00:00 UTC.

## Class: PathClippedImagePatch

**Description:** The given image is used to draw the face of the patch. Internally,
it uses BboxImage whose clippath set to the path of the patch.

FIXME : The result is currently dpi dependent.

### Function: __init__(self, path, bbox_image)

### Function: set_facecolor(self, color)

**Description:** Simply ignore facecolor.

### Function: draw(self, renderer)
