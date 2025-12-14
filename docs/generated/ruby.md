## AI Summary

A file named ruby.py.


### Function: parse_ruby(inline, m, state)

### Function: _parse_ruby_link(inline, state, pos, tokens)

### Function: render_ruby(renderer, text, rt)

### Function: ruby(md)

**Description:** A mistune plugin to support ``<ruby>`` tag. The syntax is defined
at https://lepture.com/en/2022/markdown-ruby-markup:

.. code-block:: text

    [漢字(ㄏㄢˋㄗˋ)]
    [漢(ㄏㄢˋ)字(ㄗˋ)]

    [漢字(ㄏㄢˋㄗˋ)][link]
    [漢字(ㄏㄢˋㄗˋ)](/url "title")

    [link]: /url "title"

:param md: Markdown instance
