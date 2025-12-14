## AI Summary

A file named def_list.py.


### Function: parse_def_list(block, m, state)

### Function: _parse_def_item(block, m)

### Function: _process_text(block, text, loose)

### Function: render_def_list(renderer, text)

### Function: render_def_list_head(renderer, text)

### Function: render_def_list_item(renderer, text)

### Function: def_list(md)

**Description:** A mistune plugin to support def list, spec defined at
https://michelf.ca/projects/php-markdown/extra/#def-list

Here is an example:

.. code-block:: text

    Apple
    :   Pomaceous fruit of plants of the genus Malus in
        the family Rosaceae.

    Orange
    :   The fruit of an evergreen tree of the genus Citrus.

It will be converted into HTML:

.. code-block:: html

    <dl>
    <dt>Apple</dt>
    <dd>Pomaceous fruit of plants of the genus Malus in
    the family Rosaceae.</dd>

    <dt>Orange</dt>
    <dd>The fruit of an evergreen tree of the genus Citrus.</dd>
    </dl>

:param md: Markdown instance
