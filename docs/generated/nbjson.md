## AI Summary

A file named nbjson.py.


## Class: BytesEncoder

**Description:** A JSON encoder that accepts b64 (and other *ascii*) bytestrings.

## Class: JSONReader

**Description:** A JSON notebook reader.

## Class: JSONWriter

**Description:** A JSON notebook writer.

### Function: default(self, obj)

**Description:** Get the default value of an object.

### Function: reads(self, s)

**Description:** Read a JSON string into a Notebook object

### Function: to_notebook(self, d)

**Description:** Convert a disk-format notebook dict to in-memory NotebookNode

handles multi-line values as strings, scrubbing of transient values, etc.

### Function: writes(self, nb)

**Description:** Serialize a NotebookNode object as a JSON string
