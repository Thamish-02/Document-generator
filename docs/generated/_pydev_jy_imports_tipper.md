## AI Summary

A file named _pydev_jy_imports_tipper.py.


### Function: _imp(name)

### Function: Find(name)

### Function: format_param_class_name(paramClassName)

### Function: generate_tip(data, log)

## Class: Info

### Function: isclass(cls)

### Function: ismethod(func)

**Description:** this function should return the information gathered on a function

@param func: this is the function we want to get info on
@return a tuple where:
    0 = indicates whether the parameter passed is a method or not
    1 = a list of classes 'Info', with the info gathered from the function
        this is a list because when we have methods from java with the same name and different signatures,
        we actually have many methods, each with its own set of arguments

### Function: ismodule(mod)

### Function: dir_obj(obj)

### Function: format_arg(arg)

**Description:** formats an argument to be shown

### Function: search_definition(data)

**Description:** @return file, line, col

### Function: generate_imports_tip_for_module(obj_to_complete, dir_comps, getattr, filter)

**Description:** @param obj_to_complete: the object from where we should get the completions
@param dir_comps: if passed, we should not 'dir' the object and should just iterate those passed as a parameter
@param getattr: the way to get a given object from the obj_to_complete (used for the completer)
@param filter: a callable that receives the name and decides if it should be appended or not to the results
@return: list of tuples, so that each tuple represents a completion with:
    name, doc, args, type (from the TYPE_* constants)

### Function: __init__(self, name)

### Function: basic_as_str(self)

**Description:** @returns this class information as a string (just basic format)

### Function: get_as_doc(self)

### Function: getargs(func_code)

**Description:** Get information about the arguments accepted by a code object.

Three things are returned: (args, varargs, varkw), where 'args' is
a list of argument names (possibly containing nested lists), and
'varargs' and 'varkw' are the names of the * and ** arguments or None.
