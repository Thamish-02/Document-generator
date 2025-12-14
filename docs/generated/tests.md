## AI Summary

A file named tests.py.


### Function: test_odd(value)

**Description:** Return true if the variable is odd.

### Function: test_even(value)

**Description:** Return true if the variable is even.

### Function: test_divisibleby(value, num)

**Description:** Check if a variable is divisible by a number.

### Function: test_defined(value)

**Description:** Return true if the variable is defined:

.. sourcecode:: jinja

    {% if variable is defined %}
        value of variable: {{ variable }}
    {% else %}
        variable is not defined
    {% endif %}

See the :func:`default` filter for a simple way to set undefined
variables.

### Function: test_undefined(value)

**Description:** Like :func:`defined` but the other way round.

### Function: test_filter(env, value)

**Description:** Check if a filter exists by name. Useful if a filter may be
optionally available.

.. code-block:: jinja

    {% if 'markdown' is filter %}
        {{ value | markdown }}
    {% else %}
        {{ value }}
    {% endif %}

.. versionadded:: 3.0

### Function: test_test(env, value)

**Description:** Check if a test exists by name. Useful if a test may be
optionally available.

.. code-block:: jinja

    {% if 'loud' is test %}
        {% if value is loud %}
            {{ value|upper }}
        {% else %}
            {{ value|lower }}
        {% endif %}
    {% else %}
        {{ value }}
    {% endif %}

.. versionadded:: 3.0

### Function: test_none(value)

**Description:** Return true if the variable is none.

### Function: test_boolean(value)

**Description:** Return true if the object is a boolean value.

.. versionadded:: 2.11

### Function: test_false(value)

**Description:** Return true if the object is False.

.. versionadded:: 2.11

### Function: test_true(value)

**Description:** Return true if the object is True.

.. versionadded:: 2.11

### Function: test_integer(value)

**Description:** Return true if the object is an integer.

.. versionadded:: 2.11

### Function: test_float(value)

**Description:** Return true if the object is a float.

.. versionadded:: 2.11

### Function: test_lower(value)

**Description:** Return true if the variable is lowercased.

### Function: test_upper(value)

**Description:** Return true if the variable is uppercased.

### Function: test_string(value)

**Description:** Return true if the object is a string.

### Function: test_mapping(value)

**Description:** Return true if the object is a mapping (dict etc.).

.. versionadded:: 2.6

### Function: test_number(value)

**Description:** Return true if the variable is a number.

### Function: test_sequence(value)

**Description:** Return true if the variable is a sequence. Sequences are variables
that are iterable.

### Function: test_sameas(value, other)

**Description:** Check if an object points to the same memory address than another
object:

.. sourcecode:: jinja

    {% if foo.attribute is sameas false %}
        the foo attribute really is the `False` singleton
    {% endif %}

### Function: test_iterable(value)

**Description:** Check if it's possible to iterate over an object.

### Function: test_escaped(value)

**Description:** Check if the value is escaped.

### Function: test_in(value, seq)

**Description:** Check if value is in seq.

.. versionadded:: 2.10
