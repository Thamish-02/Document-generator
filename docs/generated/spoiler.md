## AI Summary

A file named spoiler.py.


### Function: parse_block_spoiler(block, m, state)

### Function: parse_inline_spoiler(inline, m, state)

### Function: render_block_spoiler(renderer, text)

### Function: render_inline_spoiler(renderer, text)

### Function: spoiler(md)

**Description:** A mistune plugin to support block and inline spoiler. The
syntax is inspired by stackexchange:

.. code-block:: text

    Block level spoiler looks like block quote, but with `>!`:

    >! this is spoiler
    >!
    >! the content will be hidden

    Inline spoiler is surrounded by `>!` and `!<`, such as >! hide me !<.

:param md: Markdown instance
