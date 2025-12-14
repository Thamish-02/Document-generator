## AI Summary

A file named pydevd_file_utils.py.


### Function: _get_library_dir()

### Function: _resolve_listing(resolved, iter_parts_lowercase, cache)

### Function: _resolve_listing_parts(resolved, parts_in_lowercase, filename)

### Function: _normcase_linux(filename)

### Function: normcase(s, NORMCASE_CACHE)

### Function: normcase_from_client(s)

### Function: set_ide_os(os)

**Description:** We need to set the IDE os because the host where the code is running may be
actually different from the client (and the point is that we want the proper
paths to translate from the client to the server).

:param os:
    'UNIX' or 'WINDOWS'

### Function: canonical_normalized_path(filename)

**Description:** This returns a filename that is canonical and it's meant to be used internally
to store information on breakpoints and see if there's any hit on it.

Note that this version is only internal as it may not match the case and
may have symlinks resolved (and thus may not match what the user expects
in the editor).

### Function: absolute_path(filename)

**Description:** Provides a version of the filename that's absolute (and NOT normalized).

### Function: basename(filename)

**Description:** Provides the basename for a file.

### Function: _abs_and_canonical_path(filename, NORM_PATHS_CONTAINER)

### Function: _get_relative_filename_abs_path(filename, func, os_path_exists)

### Function: _apply_func_and_normalize_case(filename, func, isabs, normalize_case, os_path_exists, join)

### Function: exists(filename)

### Function: _path_to_expected_str(filename)

### Function: _original_file_to_client(filename, cache)

### Function: _original_map_file_to_server(filename)

### Function: _fix_path(path, sep, add_end_sep)

### Function: get_client_filename_source_reference(client_filename)

### Function: get_server_filename_from_source_reference(source_reference)

### Function: create_source_reference_for_linecache(server_filename)

### Function: get_source_reference_filename_from_linecache(source_reference)

### Function: create_source_reference_for_frame_id(frame_id, original_filename)

### Function: get_frame_id_from_source_reference(source_reference)

### Function: set_resolve_symlinks(resolve_symlinks)

### Function: setup_client_server_paths(paths)

**Description:** paths is the same format as PATHS_FROM_ECLIPSE_TO_PYTHON

### Function: get_abs_path_real_path_and_base_from_file(filename, NORM_PATHS_AND_BASE_CONTAINER)

### Function: get_abs_path_real_path_and_base_from_frame(frame, NORM_PATHS_AND_BASE_CONTAINER)

### Function: get_fullname(mod_name)

### Function: get_package_dir(mod_name)

### Function: _normcase_windows(filename)

### Function: _normcase_windows(filename)

### Function: _normcase_lower(filename)

### Function: _map_file_to_server(filename, cache)

### Function: _map_file_to_client(filename, cache)

### Function: _convert_to_long_pathname(filename)

### Function: _convert_to_short_pathname(filename)

### Function: _get_path_with_real_case(filename)

### Function: get_path_with_real_case(filename)

### Function: get_path_with_real_case(filename)

### Function: _normcase_lower(filename)
