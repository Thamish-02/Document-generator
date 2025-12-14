## AI Summary

A file named pydevd_xml.py.


### Function: make_valid_xml_value(s)

## Class: ExceptionOnEvaluate

### Function: _create_default_type_map()

## Class: TypeResolveHandler

### Function: is_builtin(x)

### Function: should_evaluate_full_value(val)

### Function: return_values_from_dict_to_xml(return_dict)

### Function: frame_vars_to_xml(frame_f_locals, hidden_ns)

**Description:** dumps frame variables to XML
<var name="var_name" scope="local" type="type" value="value"/>

### Function: get_variable_details(val, evaluate_full_value, to_string, context)

**Description:** :param context:
    This is the context in which the variable is being requested. Valid values:
        "watch",
        "repl",
        "hover",
        "clipboard"

### Function: var_to_xml(val, name, trim_if_too_big, additional_in_xml, evaluate_full_value)

**Description:** single variable or dictionary to xml representation

### Function: __init__(self, result, etype, tb)

### Function: __init__(self)

### Function: _initialize(self)

### Function: get_type(self, o)

### Function: _get_type(self, o, type_object, type_name)

### Function: _get_str_from_provider(self, provider, o, context)

### Function: str_from_providers(self, o, type_object, type_name, context)

### Function: _get_type(self, o, type_object, type_name)
