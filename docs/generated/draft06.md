## AI Summary

A file named draft06.py.


## Class: CodeGeneratorDraft06

### Function: __init__(self, definition, resolver, formats, use_default, use_formats, detailed_exceptions)

### Function: _generate_func_code_block(self, definition)

### Function: generate_boolean_schema(self)

**Description:** Means that schema can be specified by boolean.
True means everything is valid, False everything is invalid.

### Function: generate_type(self)

**Description:** Validation of type. Can be one type or list of types.

Since draft 06 a float without fractional part is an integer.

.. code-block:: python

    {'type': 'string'}
    {'type': ['string', 'number']}

### Function: generate_exclusive_minimum(self)

### Function: generate_exclusive_maximum(self)

### Function: generate_property_names(self)

**Description:** Means that keys of object must to follow this definition.

.. code-block:: python

    {
        'propertyNames': {
            'maxLength': 3,
        },
    }

Valid keys of object for this definition are foo, bar, ... but not foobar for example.

### Function: generate_contains(self)

**Description:** Means that array must contain at least one defined item.

.. code-block:: python

    {
        'contains': {
            'type': 'number',
        },
    }

Valid array is any with at least one number.

### Function: generate_const(self)

**Description:** Means that value is valid when is equeal to const definition.

.. code-block:: python

    {
        'const': 42,
    }

Only valid value is 42 in this example.
