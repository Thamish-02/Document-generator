## AI Summary

A file named pydevd_filtering.py.


### Function: _convert_to_str_and_clear_empty(roots)

### Function: _check_matches(patterns, paths)

### Function: glob_matches_path(path, pattern, sep, altsep)

## Class: FilesFiltering

**Description:** Note: calls at FilesFiltering are uncached.

The actual API used should be through PyDB.

### Function: __init__(self)

### Function: _get_default_library_roots(cls)

### Function: _fix_roots(self, roots)

### Function: _absolute_normalized_path(self, filename)

**Description:** Provides a version of the filename that's absolute and normalized.

### Function: set_project_roots(self, project_roots)

### Function: _get_project_roots(self)

### Function: set_library_roots(self, roots)

### Function: _get_library_roots(self)

### Function: in_project_roots(self, received_filename)

**Description:** Note: don't call directly. Use PyDb.in_project_scope (there's no caching here and it doesn't
handle all possibilities for knowing whether a project is actually in the scope, it
just handles the heuristics based on the absolute_normalized_filename without the actual frame).

### Function: use_libraries_filter(self)

**Description:** Should we debug only what's inside project folders?

### Function: set_use_libraries_filter(self, use)

### Function: use_exclude_filters(self)

### Function: exclude_by_filter(self, absolute_filename, module_name)

**Description:** :return: True if it should be excluded, False if it should be included and None
    if no rule matched the given file.

### Function: set_exclude_filters(self, exclude_filters)

**Description:** :param list(ExcludeFilter) exclude_filters:
