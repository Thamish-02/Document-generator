## AI Summary

A file named pydevd_extension_api.py.


### Function: _with_metaclass(meta)

**Description:** Create a base class with a metaclass.

## Class: _AbstractResolver

**Description:** This class exists only for documentation purposes to explain how to create a resolver.

Some examples on how to resolve things:
- list: get_dictionary could return a dict with index->item and use the index to resolve it later
- set: get_dictionary could return a dict with id(object)->object and reiterate in that array to resolve it later
- arbitrary instance: get_dictionary could return dict with attr_name->attr and use getattr to resolve it later

## Class: _AbstractProvider

## Class: TypeResolveProvider

**Description:** Implement this in an extension to provide a custom resolver, see _AbstractResolver

## Class: StrPresentationProvider

**Description:** Implement this in an extension to provide a str presentation for a type

## Class: DebuggerEventHandler

**Description:** Implement this to receive lifecycle events from the debugger

## Class: metaclass

### Function: resolve(self, var, attribute)

**Description:** In this method, we'll resolve some child item given the string representation of the item in the key
representing the previously asked dictionary.

:param var: this is the actual variable to be resolved.
:param attribute: this is the string representation of a key previously returned in get_dictionary.

### Function: get_dictionary(self, var)

**Description:** :param var: this is the variable that should have its children gotten.

:return: a dictionary where each pair key, value should be shown to the user as children items
in the variables view for the given var.

### Function: can_provide(self, type_object, type_name)

### Function: get_str_in_context(self, val, context)

**Description:** :param val:
    This is the object for which we want a string representation.

:param context:
    This is the context in which the variable is being requested. Valid values:
        "watch",
        "repl",
        "hover",
        "clipboard"

:note: this method is not required (if it's not available, get_str is called directly,
       so, it's only needed if the string representation needs to be converted based on
       the context).

### Function: get_str(self, val)

### Function: on_debugger_modules_loaded(self)

**Description:** This method invoked after all debugger modules are loaded. Useful for importing and/or patching debugger
modules at a safe time
:param kwargs: This is intended to be flexible dict passed from the debugger.
Currently passes the debugger version

### Function: __new__(cls, name, this_bases, d)
