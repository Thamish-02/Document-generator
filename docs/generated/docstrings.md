## AI Summary

A file named docstrings.py.


### Function: _get_numpy_doc_string_cls()

### Function: _search_param_in_numpydocstr(docstr, param_str)

**Description:** Search `docstr` (in numpydoc format) for type(-s) of `param_str`.

### Function: _search_return_in_numpydocstr(docstr)

**Description:** Search `docstr` (in numpydoc format) for type(-s) of function returns.

### Function: _expand_typestr(type_str)

**Description:** Attempts to interpret the possible types in `type_str`

### Function: _search_param_in_docstr(docstr, param_str)

**Description:** Search `docstr` for type(-s) of `param_str`.

>>> _search_param_in_docstr(':type param: int', 'param')
['int']
>>> _search_param_in_docstr('@type param: int', 'param')
['int']
>>> _search_param_in_docstr(
...   ':type param: :class:`threading.Thread`', 'param')
['threading.Thread']
>>> bool(_search_param_in_docstr('no document', 'param'))
False
>>> _search_param_in_docstr(':param int param: some description', 'param')
['int']

### Function: _strip_rst_role(type_str)

**Description:** Strip off the part looks like a ReST role in `type_str`.

>>> _strip_rst_role(':class:`ClassName`')  # strip off :class:
'ClassName'
>>> _strip_rst_role(':py:obj:`module.Object`')  # works with domain
'module.Object'
>>> _strip_rst_role('ClassName')  # do nothing when not ReST role
'ClassName'

See also:
http://sphinx-doc.org/domains.html#cross-referencing-python-objects

### Function: _infer_for_statement_string(module_context, string)

### Function: _execute_types_in_stmt(module_context, stmt)

**Description:** Executing all types or general elements that we find in a statement. This
doesn't include tuple, list and dict literals, because the stuff they
contain is executed. (Used as type information).

### Function: _execute_array_values(inference_state, array)

**Description:** Tuples indicate that there's not just one return value, but the listed
ones.  `(str, int)` means that it returns a tuple with both types.

### Function: infer_param(function_value, param)

### Function: infer_return_types(function_value)

### Function: infer_docstring(docstring)

### Function: search_return_in_docstr(code)
