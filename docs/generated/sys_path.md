## AI Summary

A file named sys_path.py.


### Function: _abs_path(module_context, str_path)

### Function: _paths_from_assignment(module_context, expr_stmt)

**Description:** Extracts the assigned strings from an assignment that looks as follows::

    sys.path[0:0] = ['module/path', 'another/module/path']

This function is in general pretty tolerant (and therefore 'buggy').
However, it's not a big issue usually to add more paths to Jedi's sys_path,
because it will only affect Jedi in very random situations and by adding
more paths than necessary, it usually benefits the general user.

### Function: _paths_from_list_modifications(module_context, trailer1, trailer2)

**Description:** extract the path from either "sys.path.append" or "sys.path.insert" 

### Function: check_sys_path_modifications(module_context)

**Description:** Detect sys.path modifications within module.

### Function: discover_buildout_paths(inference_state, script_path)

### Function: _get_paths_from_buildout_script(inference_state, buildout_script_path)

### Function: _get_parent_dir_with_file(path, filename)

### Function: _get_buildout_script_paths(search_path)

**Description:** if there is a 'buildout.cfg' file in one of the parent directories of the
given module it will return a list of all files in the buildout bin
directory that look like python files.

:param search_path: absolute path to the module.

### Function: remove_python_path_suffix(path)

### Function: transform_path_to_dotted(sys_path, module_path)

**Description:** Returns the dotted path inside a sys.path as a list of names. e.g.

>>> transform_path_to_dotted([str(Path("/foo").absolute())], Path('/foo/bar/baz.py').absolute())
(('bar', 'baz'), False)

Returns (None, False) if the path doesn't really resolve to anything.
The second return part is if it is a package.

### Function: get_sys_path_powers(names)

### Function: iter_potential_solutions()
