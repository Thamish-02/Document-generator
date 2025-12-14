## AI Summary

A file named table.py.


### Function: parse_table(block, m, state)

### Function: parse_nptable(block, m, state)

### Function: _process_thead(header, align)

### Function: _process_row(text, aligns)

### Function: render_table(renderer, text)

### Function: render_table_head(renderer, text)

### Function: render_table_body(renderer, text)

### Function: render_table_row(renderer, text)

### Function: render_table_cell(renderer, text, align, head)

### Function: table(md)

**Description:** A mistune plugin to support table, spec defined at
https://michelf.ca/projects/php-markdown/extra/#table

Here is an example:

.. code-block:: text

    First Header  | Second Header
    ------------- | -------------
    Content Cell  | Content Cell
    Content Cell  | Content Cell

:param md: Markdown instance

### Function: table_in_quote(md)

**Description:** Enable table plugin in block quotes.

### Function: table_in_list(md)

**Description:** Enable table plugin in list.
