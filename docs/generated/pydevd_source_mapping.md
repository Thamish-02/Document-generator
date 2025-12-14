## AI Summary

A file named pydevd_source_mapping.py.


## Class: SourceMappingEntry

## Class: SourceMapping

### Function: __init__(self, line, end_line, runtime_line, runtime_source)

### Function: contains_line(self, i)

### Function: contains_runtime_line(self, i)

### Function: __str__(self)

### Function: __init__(self, on_source_mapping_changed)

### Function: set_source_mapping(self, absolute_filename, mapping)

**Description:** :param str absolute_filename:
    The filename for the source mapping (bytes on py2 and str on py3).

:param list(SourceMappingEntry) mapping:
    A list with the source mapping entries to be applied to the given filename.

:return str:
    An error message if it was not possible to set the mapping or an empty string if
    everything is ok.

### Function: map_to_client(self, runtime_source_filename, lineno)

### Function: has_mapping_entry(self, runtime_source_filename)

**Description:** :param runtime_source_filename:
    Something as <ipython-cell-xxx>

### Function: map_to_server(self, absolute_filename, lineno)

**Description:** Convert something as 'file1.py' at line 10 to '<ipython-cell-xxx>' at line 2.

Note that the name should be already normalized at this point.
