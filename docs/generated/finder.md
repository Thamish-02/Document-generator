## AI Summary

A file named finder.py.


### Function: filter_name(filters, name_or_str)

**Description:** Searches names that are defined in a scope (the different
``filters``), until a name fits.

### Function: _remove_del_stmt(names)

### Function: check_flow_information(value, flow, search_name, pos)

**Description:** Try to find out the type of a variable just with the information that
is given by the flows: e.g. It is also responsible for assert checks.::

    if isinstance(k, str):
        k.  # <- completion here

ensures that `k` is a string.

### Function: _get_isinstance_trailer_arglist(node)

### Function: _check_isinstance_type(value, node, search_name)

### Function: _get_call_string(node)
