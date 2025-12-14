## AI Summary

A file named roles.py.


## Class: _QueryReference

**Description:** Wraps a reference or pending reference to add a query string.

The query string is generated from the attributes added to this node.

Also equivalent to a `~docutils.nodes.literal` node.

### Function: _visit_query_reference_node(self, node)

**Description:** Resolve *node* into query strings on its ``reference`` children.

Then act as if this is a `~docutils.nodes.literal`.

### Function: _depart_query_reference_node(self, node)

**Description:** Act as if this is a `~docutils.nodes.literal`.

### Function: _rcparam_role(name, rawtext, text, lineno, inliner, options, content)

**Description:** Sphinx role ``:rc:`` to highlight and link ``rcParams`` entries.

Usage: Give the desired ``rcParams`` key as parameter.

:code:`:rc:`figure.dpi`` will render as: :rc:`figure.dpi`

### Function: _mpltype_role(name, rawtext, text, lineno, inliner, options, content)

**Description:** Sphinx role ``:mpltype:`` for custom matplotlib types.

In Matplotlib, there are a number of type-like concepts that do not have a
direct type representation; example: color. This role allows to properly
highlight them in the docs and link to their definition.

Currently supported values:

- :code:`:mpltype:`color`` will render as: :mpltype:`color`

### Function: setup(app)

### Function: to_query_string(self)

**Description:** Generate query string from node attributes.
