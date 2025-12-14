## AI Summary

A file named generator.py.


### Function: enforce_list(variable)

## Class: CodeGenerator

**Description:** This class is not supposed to be used directly. Anything
inside of this class can be changed without noticing.

This class generates code of validation function from JSON
schema object as string. Example:

.. code-block:: python

    CodeGenerator(json_schema_definition).func_code

### Function: serialize_regexes(patterns_dict)

### Function: repr_regex(regex)

### Function: __init__(self, definition, resolver, detailed_exceptions)

### Function: func_code(self)

**Description:** Returns generated code of whole validation function as string.

### Function: global_state(self)

**Description:** Returns global variables for generating function from ``func_code``. Includes
compiled regular expressions and imports, so it does not have to do it every
time when validation function is called.

### Function: global_state_code(self)

**Description:** Returns global variables for generating function from ``func_code`` as code.
Includes compiled regular expressions and imports.

### Function: _generate_func_code(self)

### Function: generate_func_code(self)

**Description:** Creates base code of validation function and calls helper
for creating code by definition.

### Function: generate_validation_function(self, uri, name)

**Description:** Generate validation function for given uri with given name

### Function: generate_func_code_block(self, definition, variable, variable_name, clear_variables)

**Description:** Creates validation rules for current definition.

Returns the number of validation rules generated as code.

### Function: _generate_func_code_block(self, definition)

### Function: run_generate_functions(self, definition)

**Description:** Returns the number of generate functions that were executed.

### Function: generate_ref(self)

**Description:** Ref can be link to remote or local definition.

.. code-block:: python

    {'$ref': 'http://json-schema.org/draft-04/schema#'}
    {
        'properties': {
            'foo': {'type': 'integer'},
            'bar': {'$ref': '#/properties/foo'}
        }
    }

### Function: l(self, line)

**Description:** Short-cut of line. Used for inserting line. It's formated with parameters
``variable``, ``variable_name`` (as ``name`` for short-cut), all keys from
current JSON schema ``definition`` and also passed arguments in ``args``
and named ``kwds``.

.. code-block:: python

    self.l('if {variable} not in {enum}: raise JsonSchemaValueException("Wrong!")')

When you want to indent block, use it as context manager. For example:

.. code-block:: python

    with self.l('if {variable} not in {enum}:'):
        self.l('raise JsonSchemaValueException("Wrong!")')

### Function: e(self, string)

**Description:** Short-cut of escape. Used for inserting user values into a string message.

.. code-block:: python

    self.l('raise JsonSchemaValueException("Variable: {}")', self.e(variable))

### Function: exc(self, msg)

**Description:** Short-cut for creating raising exception in the code.

### Function: _expand_refs(self, definition)

### Function: create_variable_with_length(self)

**Description:** Append code for creating variable with length of that variable
(for example length of list or dictionary) with name ``{variable}_len``.
It can be called several times and always it's done only when that variable
still does not exists.

### Function: create_variable_keys(self)

**Description:** Append code for creating variable with keys of that variable (dictionary)
with a name ``{variable}_keys``. Similar to `create_variable_with_length`.

### Function: create_variable_is_list(self)

**Description:** Append code for creating variable with bool if it's instance of list
with a name ``{variable}_is_list``. Similar to `create_variable_with_length`.

### Function: create_variable_is_dict(self)

**Description:** Append code for creating variable with bool if it's instance of list
with a name ``{variable}_is_dict``. Similar to `create_variable_with_length`.
