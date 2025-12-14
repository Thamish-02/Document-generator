## AI Summary

A file named tree.py.


## Class: Meta

## Class: Tree

**Description:** The main tree class.

Creates a new tree, and stores "data" and "children" in attributes of the same name.
Trees can be hashed and compared.

Parameters:
    data: The name of the rule or alias
    children: List of matched sub-rules and terminals
    meta: Line & Column numbers (if ``propagate_positions`` is enabled).
        meta attributes: (line, column, end_line, end_column, start_pos, end_pos,
                          container_line, container_column, container_end_line, container_end_column)
        container_* attributes consider all symbols, including those that have been inlined in the tree.
        For example, in the rule 'a: _A B _C', the regular attributes will mark the start and end of B,
        but the container_* attributes will also include _A and _C in the range. However, rules that
        contain 'a' will consider it in full, including _A and _C for all attributes.

## Class: SlottedTree

### Function: pydot__tree_to_png(tree, filename, rankdir)

### Function: pydot__tree_to_dot(tree, filename, rankdir)

### Function: pydot__tree_to_graph(tree, rankdir)

**Description:** Creates a colorful image that represents the tree (data+children, without meta)

Possible values for `rankdir` are "TB", "LR", "BT", "RL", corresponding to
directed graphs drawn from top to bottom, from left to right, from bottom to
top, and from right to left, respectively.

`kwargs` can be any graph attribute (e. g. `dpi=200`). For a list of
possible attributes, see https://www.graphviz.org/doc/info/attrs.html.

### Function: __init__(self)

### Function: __init__(self, data, children, meta)

### Function: meta(self)

### Function: __repr__(self)

### Function: _pretty_label(self)

### Function: _pretty(self, level, indent_str)

### Function: pretty(self, indent_str)

**Description:** Returns an indented string representation of the tree.

Great for debugging.

### Function: __rich__(self, parent)

**Description:** Returns a tree widget for the 'rich' library.

Example:
    ::
        from rich import print
        from lark import Tree

        tree = Tree('root', ['node1', 'node2'])
        print(tree)

### Function: _rich(self, parent)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __hash__(self)

### Function: iter_subtrees(self)

**Description:** Depth-first iteration.

Iterates over all the subtrees, never returning to the same node twice (Lark's parse-tree is actually a DAG).

### Function: iter_subtrees_topdown(self)

**Description:** Breadth-first iteration.

Iterates over all the subtrees, return nodes in order like pretty() does.

### Function: find_pred(self, pred)

**Description:** Returns all nodes of the tree that evaluate pred(node) as true.

### Function: find_data(self, data)

**Description:** Returns all nodes of the tree whose data equals the given data.

### Function: find_token(self, token_type)

**Description:** Returns all tokens whose type equals the given token_type.

This is a recursive function that will find tokens in all the subtrees.

Example:
    >>> term_tokens = tree.find_token('TERM')

### Function: expand_kids_by_data(self)

**Description:** Expand (inline) children with any of the given data values. Returns True if anything changed

### Function: scan_values(self, pred)

**Description:** Return all values in the tree that evaluate pred(value) as true.

This can be used to find all the tokens in the tree.

Example:
    >>> all_tokens = tree.scan_values(lambda v: isinstance(v, Token))

### Function: __deepcopy__(self, memo)

### Function: copy(self)

### Function: set(self, data, children)

### Function: new_leaf(leaf)

### Function: _to_pydot(subtree)
