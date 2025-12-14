## AI Summary

A file named compare.py.


### Function: make_test_filename(fname, purpose)

**Description:** Make a new filename by inserting *purpose* before the file's extension.

### Function: _get_cache_path()

### Function: get_cache_dir()

### Function: get_file_hash(path, block_size)

## Class: _ConverterError

## Class: _Converter

## Class: _GSConverter

## Class: _SVGConverter

## Class: _SVGWithMatplotlibFontsConverter

**Description:** A SVG converter which explicitly adds the fonts shipped by Matplotlib to
Inkspace's font search path, to better support `svg.fonttype = "none"`
(which is in particular used by certain mathtext tests).

### Function: _update_converter()

### Function: comparable_formats()

**Description:** Return the list of file formats that `.compare_images` can compare
on this system.

Returns
-------
list of str
    E.g. ``['png', 'pdf', 'svg', 'eps']``.

### Function: convert(filename, cache)

**Description:** Convert the named file to png; return the name of the created file.

If *cache* is True, the result of the conversion is cached in
`matplotlib.get_cachedir() + '/test_cache/'`.  The caching is based on a
hash of the exact contents of the input file.  Old cache entries are
automatically deleted as needed to keep the size of the cache capped to
twice the size of all baseline images.

### Function: _clean_conversion_cache()

### Function: _register_conversion_cache_cleaner_once()

### Function: crop_to_same(actual_path, actual_image, expected_path, expected_image)

### Function: calculate_rms(expected_image, actual_image)

**Description:** Calculate the per-pixel errors, then compute the root mean square error.

### Function: _load_image(path)

### Function: compare_images(expected, actual, tol, in_decorator)

**Description:** Compare two "image" files checking differences within a tolerance.

The two given filenames may point to files which are convertible to
PNG via the `.converter` dictionary. The underlying RMS is calculated
with the `.calculate_rms` function.

Parameters
----------
expected : str
    The filename of the expected image.
actual : str
    The filename of the actual image.
tol : float
    The tolerance (a color value difference, where 255 is the
    maximal difference).  The test fails if the average pixel
    difference is greater than this value.
in_decorator : bool
    Determines the output format. If called from image_comparison
    decorator, this should be True. (default=False)

Returns
-------
None or dict or str
    Return *None* if the images are equal within the given tolerance.

    If the images differ, the return value depends on  *in_decorator*.
    If *in_decorator* is true, a dict with the following entries is
    returned:

    - *rms*: The RMS of the image difference.
    - *expected*: The filename of the expected image.
    - *actual*: The filename of the actual image.
    - *diff_image*: The filename of the difference image.
    - *tol*: The comparison tolerance.

    Otherwise, a human-readable multi-line string representation of this
    information is returned.

Examples
--------
::

    img1 = "./baseline/plot.png"
    img2 = "./output/plot.png"
    compare_images(img1, img2, 0.001)

### Function: save_diff_image(expected, actual, output)

**Description:** Parameters
----------
expected : str
    File path of expected image.
actual : str
    File path of actual image.
output : str
    File path to save difference image to.

### Function: __init__(self)

### Function: __del__(self)

### Function: _read_until(self, terminator)

**Description:** Read until the prompt is reached.

### Function: __call__(self, orig, dest)

### Function: __call__(self, orig, dest)

### Function: __del__(self)

### Function: __call__(self, orig, dest)

### Function: encode_and_escape(name)
