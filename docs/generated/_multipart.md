## AI Summary

A file named _multipart.py.


### Function: _format_form_param(name, value)

**Description:** Encode a name/value pair within a multipart form.

### Function: _guess_content_type(filename)

**Description:** Guesses the mimetype based on a filename. Defaults to `application/octet-stream`.

Returns `None` if `filename` is `None` or empty.

### Function: get_multipart_boundary_from_content_type(content_type)

## Class: DataField

**Description:** A single form field item, within a multipart form field.

## Class: FileField

**Description:** A single file field item, within a multipart form field.

## Class: MultipartStream

**Description:** Request content as streaming multipart encoded form data.

### Function: replacer(match)

### Function: __init__(self, name, value)

### Function: render_headers(self)

### Function: render_data(self)

### Function: get_length(self)

### Function: render(self)

### Function: __init__(self, name, value)

### Function: get_length(self)

### Function: render_headers(self)

### Function: render_data(self)

### Function: render(self)

### Function: __init__(self, data, files, boundary)

### Function: _iter_fields(self, data, files)

### Function: iter_chunks(self)

### Function: get_content_length(self)

**Description:** Return the length of the multipart encoded content, or `None` if
any of the files have a length that cannot be determined upfront.

### Function: get_headers(self)

### Function: __iter__(self)
