## AI Summary

A file named widget_media.py.


## Class: _Media

**Description:** Base class for Image, Audio and Video widgets.

The `value` of this widget accepts a byte string.  The byte string is the
raw data that you want the browser to display.

If you pass `"url"` to the `"format"` trait, `value` will be interpreted
as a URL as bytes encoded in UTF-8.

## Class: Image

**Description:** Displays an image as a widget.

The `value` of this widget accepts a byte string.  The byte string is the
raw image data that you want the browser to display.  You can explicitly
define the format of the byte string using the `format` trait (which
defaults to "png").

If you pass `"url"` to the `"format"` trait, `value` will be interpreted
as a URL as bytes encoded in UTF-8.

## Class: Video

**Description:** Displays a video as a widget.

The `value` of this widget accepts a byte string.  The byte string is the
raw video data that you want the browser to display.  You can explicitly
define the format of the byte string using the `format` trait (which
defaults to "mp4").

If you pass `"url"` to the `"format"` trait, `value` will be interpreted
as a URL as bytes encoded in UTF-8.

## Class: Audio

**Description:** Displays a audio as a widget.

The `value` of this widget accepts a byte string.  The byte string is the
raw audio data that you want the browser to display.  You can explicitly
define the format of the byte string using the `format` trait (which
defaults to "mp3").

If you pass `"url"` to the `"format"` trait, `value` will be interpreted
as a URL as bytes encoded in UTF-8.

### Function: _from_file(cls, tag, filename)

**Description:** Create an :class:`Media` from a local file.

Parameters
----------
filename: str
    The location of a file to read into the value from disk.

**kwargs:
    The keyword arguments for `Media`

Returns an `Media` with the value set from the filename.

### Function: from_url(cls, url)

**Description:** Create an :class:`Media` from a URL.

:code:`Media.from_url(url)` is equivalent to:

.. code-block: python

    med = Media(value=url, format='url')

But both unicode and bytes arguments are allowed for ``url``.

Parameters
----------
url: [str, bytes]
    The location of a URL to load.

### Function: set_value_from_file(self, filename)

**Description:** Convenience method for reading a file into `value`.

Parameters
----------
filename: str
    The location of a file to read into value from disk.

### Function: _load_file_value(cls, filename)

### Function: _guess_format(cls, tag, filename)

### Function: _get_repr(self, cls)

### Function: __init__(self)

### Function: from_file(cls, filename)

### Function: __repr__(self)

### Function: from_file(cls, filename)

### Function: __repr__(self)

### Function: from_file(cls, filename)

### Function: __repr__(self)
