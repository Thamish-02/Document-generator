## AI Summary

A file named references.py.


### Function: _resolve_names(definition_names, avoid_names)

### Function: _dictionarize(names)

### Function: _find_defining_names(module_context, tree_name)

### Function: _find_names(module_context, tree_name)

### Function: _add_names_in_same_context(context, string_name)

### Function: _find_global_variables(names, search_name)

### Function: find_references(module_context, tree_name, only_in_module)

### Function: _check_fs(inference_state, file_io, regex)

### Function: gitignored_paths(folder_io, file_io)

### Function: expand_relative_ignore_paths(folder_io, relative_paths)

### Function: recurse_find_python_folders_and_files(folder_io, except_paths)

### Function: recurse_find_python_files(folder_io, except_paths)

### Function: _find_python_files_in_sys_path(inference_state, module_contexts)

### Function: _find_project_modules(inference_state, module_contexts)

### Function: get_module_contexts_containing_name(inference_state, module_contexts, name, limit_reduction)

**Description:** Search a name in the directories of modules.

:param limit_reduction: Divides the limits on opening/parsing files by this
    factor.

### Function: search_in_file_ios(inference_state, file_io_iterator, name, limit_reduction, complete)
