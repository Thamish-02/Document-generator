## AI Summary

A file named typeshed.py.


### Function: _merge_create_stub_map(path_infos)

### Function: _create_stub_map(directory_path_info)

**Description:** Create a mapping of an importable name in Python to a stub file.

### Function: _get_typeshed_directories(version_info)

### Function: _cache_stub_file_map(version_info)

**Description:** Returns a map of an importable name in Python to a stub file.

### Function: import_module_decorator(func)

### Function: try_to_load_stub_cached(inference_state, import_names)

### Function: _try_to_load_stub(inference_state, import_names, python_value_set, parent_module_value, sys_path)

**Description:** Trying to load a stub for a set of import_names.

This is modelled to work like "PEP 561 -- Distributing and Packaging Type
Information", see https://www.python.org/dev/peps/pep-0561.

### Function: _load_from_typeshed(inference_state, python_value_set, parent_module_value, import_names)

### Function: _try_to_load_stub_from_file(inference_state, python_value_set, file_io, import_names)

### Function: parse_stub_module(inference_state, file_io)

### Function: create_stub_module(inference_state, grammar, python_value_set, stub_module_node, file_io, import_names)

### Function: generate()

### Function: wrapper(inference_state, import_names, parent_module_value, sys_path, prefer_stubs)
