## AI Summary

A file named imports.py.


## Class: ModuleCache

### Function: infer_import(context, tree_name)

### Function: goto_import(context, tree_name)

### Function: _prepare_infer_import(module_context, tree_name)

### Function: _add_error(value, name, message)

### Function: _level_to_base_import_path(project_path, directory, level)

**Description:** In case the level is outside of the currently known package (something like
import .....foo), we can still try our best to help the user for
completions.

## Class: Importer

### Function: import_module_by_names(inference_state, import_names, sys_path, module_context, prefer_stubs)

### Function: import_module(inference_state, import_names, parent_module_value, sys_path)

**Description:** This method is very similar to importlib's `_gcd_import`.

### Function: _load_python_module(inference_state, file_io, import_names, is_package)

### Function: _load_builtin_module(inference_state, import_names, sys_path)

### Function: load_module_from_path(inference_state, file_io, import_names, is_package)

**Description:** This should pretty much only be used for get_modules_containing_name. It's
here to ensure that a random path is still properly loaded into the Jedi
module structure.

### Function: load_namespace_from_path(inference_state, folder_io)

### Function: follow_error_node_imports_if_possible(context, name)

### Function: iter_module_names(inference_state, module_context, search_path, module_cls, add_builtin_modules)

**Description:** Get the names of all modules in the search_path. This means file names
and not names defined in the files.

### Function: __init__(self)

### Function: add(self, string_names, value_set)

### Function: get(self, string_names)

### Function: __init__(self, inference_state, import_path, module_context, level)

**Description:** An implementation similar to ``__import__``. Use `follow`
to actually follow the imports.

*level* specifies whether to use absolute or relative imports. 0 (the
default) means only perform absolute imports. Positive values for level
indicate the number of parent directories to search relative to the
directory of the module calling ``__import__()`` (see PEP 328 for the
details).

:param import_path: List of namespaces (strings or Names).

### Function: _str_import_path(self)

**Description:** Returns the import path as pure strings instead of `Name`.

### Function: _sys_path_with_modifications(self, is_completion)

### Function: follow(self)

### Function: _get_module_names(self, search_path, in_module)

**Description:** Get the names of all modules in the search_path. This means file names
and not names defined in the files.

### Function: completion_names(self, inference_state, only_modules)

**Description:** :param only_modules: Indicates wheter it's possible to import a
    definition that is not defined in a module.
