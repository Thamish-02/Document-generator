## AI Summary

A file named math.py.


### Function: parse_block_math(block, m, state)

### Function: parse_inline_math(inline, m, state)

### Function: render_block_math(renderer, text)

### Function: render_inline_math(renderer, text)

### Function: math(md)

**Description:** A mistune plugin to support math. The syntax is used
by many markdown extensions:

.. code-block:: text

    Block math is surrounded by $$:

    $$
    f(a)=f(b)
    $$

    Inline math is surrounded by `$`, such as $f(a)=f(b)$

:param md: Markdown instance

### Function: math_in_quote(md)

**Description:** Enable block math plugin in block quote.

### Function: math_in_list(md)

**Description:** Enable block math plugin in list.
