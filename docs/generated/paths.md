## AI Summary

A file named paths.py.


### Function: normalized_uri(root_dir)

**Description:** Attempt to make an LSP rootUri from a ContentsManager root_dir

Special care must be taken around windows paths: the canonical form of
windows drives and UNC paths is lower case

### Function: file_uri_to_path(file_uri)

**Description:** Return a path string for give file:/// URI.

Respect the different path convention on Windows.
Based on https://stackoverflow.com/a/57463161/6646912, BSD 0

### Function: is_relative(root, path)

**Description:** Return if path is relative to root
