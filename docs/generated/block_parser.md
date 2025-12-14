## AI Summary

A file named block_parser.py.


## Class: BlockParser

### Function: _parse_html_to_end(state, end_marker, start_pos)

### Function: _parse_html_to_newline(state, newline)

### Function: __init__(self, block_quote_rules, list_rules, max_nested_level)

### Function: parse_blank_line(self, m, state)

**Description:** Parse token for blank lines.

### Function: parse_thematic_break(self, m, state)

**Description:** Parse token for thematic break, e.g. ``<hr>`` tag in HTML.

### Function: parse_indent_code(self, m, state)

**Description:** Parse token for code block which is indented by 4 spaces.

### Function: parse_fenced_code(self, m, state)

**Description:** Parse token for fenced code block. A fenced code block is started with
3 or more backtick(`) or tilde(~).

An example of a fenced code block:

.. code-block:: markdown

    ```python
    def markdown(text):
        return mistune.html(text)
    ```

### Function: parse_atx_heading(self, m, state)

**Description:** Parse token for ATX heading. An ATX heading is started with 1 to 6
symbol of ``#``.

### Function: parse_setex_heading(self, m, state)

**Description:** Parse token for setex style heading. A setex heading syntax looks like:

.. code-block:: markdown

    H1 title
    ========

### Function: parse_ref_link(self, m, state)

**Description:** Parse link references and save the link information into ``state.env``.

Here is an example of a link reference:

.. code-block:: markdown

    a [link][example]

    [example]: https://example.com "Optional title"

This method will save the link reference into ``state.env`` as::

    state.env['ref_links']['example'] = {
        'url': 'https://example.com',
        'title': "Optional title",
    }

### Function: extract_block_quote(self, m, state)

**Description:** Extract text and cursor end position of a block quote.

### Function: parse_block_quote(self, m, state)

**Description:** Parse token for block quote. Here is an example of the syntax:

.. code-block:: markdown

    > a block quote starts
    > with right arrows

### Function: parse_list(self, m, state)

**Description:** Parse tokens for ordered and unordered list.

### Function: parse_block_html(self, m, state)

### Function: parse_raw_html(self, m, state)

### Function: parse(self, state, rules)
