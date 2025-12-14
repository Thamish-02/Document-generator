## AI Summary

A file named visitor.py.


## Class: NodeVisitor

**Description:** Walks the abstract syntax tree and call visitor functions for every
node found.  The visitor functions may return values which will be
forwarded by the `visit` method.

Per default the visitor functions for the nodes are ``'visit_'`` +
class name of the node.  So a `TryFinally` node visit function would
be `visit_TryFinally`.  This behavior can be changed by overriding
the `get_visitor` function.  If no visitor function exists for a node
(return value `None`) the `generic_visit` visitor is used instead.

## Class: NodeTransformer

**Description:** Walks the abstract syntax tree and allows modifications of nodes.

The `NodeTransformer` will walk the AST and use the return value of the
visitor functions to replace or remove the old node.  If the return
value of the visitor function is `None` the node will be removed
from the previous location otherwise it's replaced with the return
value.  The return value may be the original node in which case no
replacement takes place.

## Class: VisitCallable

### Function: get_visitor(self, node)

**Description:** Return the visitor function for this node or `None` if no visitor
exists for this node.  In that case the generic visit function is
used instead.

### Function: visit(self, node)

**Description:** Visit a node.

### Function: generic_visit(self, node)

**Description:** Called if no explicit visitor function exists for a node.

### Function: generic_visit(self, node)

### Function: visit_list(self, node)

**Description:** As transformers may return lists in some places this method
can be used to enforce a list as return value.

### Function: __call__(self, node)
