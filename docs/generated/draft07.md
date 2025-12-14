## AI Summary

A file named draft07.py.


## Class: CodeGeneratorDraft07

### Function: __init__(self, definition, resolver, formats, use_default, use_formats, detailed_exceptions)

### Function: generate_if_then_else(self)

**Description:** Implementation of if-then-else.

.. code-block:: python

    {
        'if': {
            'exclusiveMaximum': 0,
        },
        'then': {
            'minimum': -10,
        },
        'else': {
            'multipleOf': 2,
        },
    }

Valid values are any between -10 and 0 or any multiplication of two.

### Function: generate_content_encoding(self)

**Description:** Means decoding value when it's encoded by base64.

.. code-block:: python

    {
        'contentEncoding': 'base64',
    }

### Function: generate_content_media_type(self)

**Description:** Means loading value when it's specified as JSON.

.. code-block:: python

    {
        'contentMediaType': 'application/json',
    }
