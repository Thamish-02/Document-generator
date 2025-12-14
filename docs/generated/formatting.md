## AI Summary

A file named formatting.py.


### Function: parse_strikethrough(inline, m, state)

### Function: render_strikethrough(renderer, text)

### Function: parse_mark(inline, m, state)

### Function: render_mark(renderer, text)

### Function: parse_insert(inline, m, state)

### Function: render_insert(renderer, text)

### Function: parse_superscript(inline, m, state)

### Function: render_superscript(renderer, text)

### Function: parse_subscript(inline, m, state)

### Function: render_subscript(renderer, text)

### Function: _parse_to_end(inline, m, state, tok_type, end_pattern)

### Function: _parse_script(inline, m, state, tok_type)

### Function: strikethrough(md)

**Description:** A mistune plugin to support strikethrough. Spec defined by
GitHub flavored Markdown and commonly used by many parsers:

.. code-block:: text

    ~~This was mistaken text~~

It will be converted into HTML:

.. code-block:: html

    <del>This was mistaken text</del>

:param md: Markdown instance

### Function: mark(md)

**Description:** A mistune plugin to add ``<mark>`` tag. Spec defined at
https://facelessuser.github.io/pymdown-extensions/extensions/mark/:

.. code-block:: text

    ==mark me== ==mark \=\= equal==

:param md: Markdown instance

### Function: insert(md)

**Description:** A mistune plugin to add ``<ins>`` tag. Spec defined at
https://facelessuser.github.io/pymdown-extensions/extensions/caret/#insert:

.. code-block:: text

    ^^insert me^^

:param md: Markdown instance

### Function: superscript(md)

**Description:** A mistune plugin to add ``<sup>`` tag. Spec defined at
https://pandoc.org/MANUAL.html#superscripts-and-subscripts:

.. code-block:: text

    2^10^ is 1024.

:param md: Markdown instance

### Function: subscript(md)

**Description:** A mistune plugin to add ``<sub>`` tag. Spec defined at
https://pandoc.org/MANUAL.html#superscripts-and-subscripts:

.. code-block:: text

    H~2~O is a liquid.

:param md: Markdown instance
