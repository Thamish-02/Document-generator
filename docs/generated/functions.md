## AI Summary

A file named functions.py.


### Function: get_sys_path()

### Function: load_module(inference_state)

### Function: get_compiled_method_return(inference_state, id, attribute)

### Function: create_simple_object(inference_state, obj)

### Function: get_module_info(inference_state, sys_path, full_name)

**Description:** Returns Tuple[Union[NamespaceInfo, FileIO, None], Optional[bool]]

### Function: get_builtin_module_names(inference_state)

### Function: _test_raise_error(inference_state, exception_type)

**Description:** Raise an error to simulate certain problems for unit tests.

### Function: _test_print(inference_state, stderr, stdout)

**Description:** Force some prints in the subprocesses. This exists for unit tests.

### Function: _get_init_path(directory_path)

**Description:** The __init__ file can be searched in a directory. If found return it, else
None.

### Function: safe_literal_eval(inference_state, value)

### Function: iter_module_names()

### Function: _iter_module_names(inference_state, paths)

### Function: _find_module(string, path, full_name, is_global_search)

**Description:** Provides information about a module.

This function isolates the differences in importing libraries introduced with
python 3.3 on; it gets a module name and optionally a path. It will return a
tuple containin an open file for the module (if not builtin), the filename
or the name of the module if it is a builtin one and a boolean indicating
if the module is contained in a package.

### Function: _find_module_py33(string, path, loader, full_name, is_global_search)

### Function: _from_loader(loader, string)

### Function: _get_source(loader, fullname)

**Description:** This method is here as a replacement for SourceLoader.get_source. That
method returns unicode, but we prefer bytes.

### Function: _zip_list_subdirectory(zip_path, zip_subdir_path)

## Class: ImplicitNSInfo

**Description:** Stores information returned from an implicit namespace spec

### Function: __init__(self, name, paths)
