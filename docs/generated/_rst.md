## AI Summary

A file named _rst.py.


## Class: RSTParser

## Class: RSTDirective

**Description:** A RST style of directive syntax is inspired by reStructuredText.
The syntax is very powerful that you can define a lot of custom
features on your own. The syntax looks like:

.. code-block:: text

    .. directive-type:: directive value
       :option-key: option value
       :option-key: option value

       content text here

To use ``RSTDirective``, developers can add it into plugin list in
the :class:`Markdown` instance:

.. code-block:: python

    import mistune
    from mistune.directives import RSTDirective, Admonition

    md = mistune.create_markdown(plugins=[
        # ...
        RSTDirective([Admonition()]),
    ])

### Function: parse_type(m)

### Function: parse_title(m)

### Function: parse_content(m)

### Function: parse_directive(self, block, m, state)

### Function: __call__(self, markdown)
