## AI Summary

A file named rwbase.py.


### Function: _is_json_mime(mime)

**Description:** Is a key a JSON mime-type that should be left alone?

### Function: _rejoin_mimebundle(data)

**Description:** Rejoin the multi-line string fields in a mimebundle (in-place)

### Function: rejoin_lines(nb)

**Description:** rejoin multiline text into strings

For reversing effects of ``split_lines(nb)``.

This only rejoins lines that have been split, so if text objects were not split
they will pass through unchanged.

Used when reading JSON files that may have been passed through split_lines.

### Function: _split_mimebundle(data)

**Description:** Split multi-line string fields in a mimebundle (in-place)

### Function: split_lines(nb)

**Description:** split likely multiline text into lists of strings

For file output more friendly to line-based VCS. ``rejoin_lines(nb)`` will
reverse the effects of ``split_lines(nb)``.

Used when writing JSON files.

### Function: strip_transient(nb)

**Description:** Strip transient values that shouldn't be stored in files.

This should be called in *both* read and write.

## Class: NotebookReader

**Description:** A class for reading notebooks.

## Class: NotebookWriter

**Description:** A class for writing notebooks.

### Function: reads(self, s)

**Description:** Read a notebook from a string.

### Function: read(self, fp)

**Description:** Read a notebook from a file like object

### Function: writes(self, nb)

**Description:** Write a notebook to a string.

### Function: write(self, nb, fp)

**Description:** Write a notebook to a file like object
