## AI Summary

A file named _pydev_imports_tipper.py.


### Function: getargspec()

### Function: _imp(name, log)

### Function: get_file(mod)

### Function: Find(name, log)

### Function: search_definition(data)

**Description:** @return file, line, col

### Function: generate_tip(data, log)

### Function: check_char(c)

### Function: generate_imports_tip_for_module(obj_to_complete, dir_comps, getattr, filter)

**Description:** @param obj_to_complete: the object from where we should get the completions
@param dir_comps: if passed, we should not 'dir' the object and should just iterate those passed as kwonly_arg parameter
@param getattr: the way to get kwonly_arg given object from the obj_to_complete (used for the completer)
@param filter: kwonly_arg callable that receives the name and decides if it should be appended or not to the results
@return: list of tuples, so that each tuple represents kwonly_arg completion with:
    name, doc, args, type (from the TYPE_* constants)

### Function: signature_from_docstring(doc, obj_name)

### Function: _imp(name, log)
