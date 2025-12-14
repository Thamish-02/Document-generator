## AI Summary

A file named earley_forest.py.


## Class: ForestNode

## Class: SymbolNode

**Description:** A Symbol Node represents a symbol (or Intermediate LR0).

Symbol nodes are keyed by the symbol (s). For intermediate nodes
s will be an LR0, stored as a tuple of (rule, ptr). For completed symbol
nodes, s will be a string representing the non-terminal origin (i.e.
the left hand side of the rule).

The children of a Symbol or Intermediate Node will always be Packed Nodes;
with each Packed Node child representing a single derivation of a production.

Hence a Symbol Node with a single child is unambiguous.

Parameters:
    s: A Symbol, or a tuple of (rule, ptr) for an intermediate node.
    start: For dynamic lexers, the index of the start of the substring matched by this symbol (inclusive).
    end: For dynamic lexers, the index of the end of the substring matched by this symbol (exclusive).

Properties:
    is_intermediate: True if this node is an intermediate node.
    priority: The priority of the node's symbol.

## Class: StableSymbolNode

**Description:** A version of SymbolNode that uses OrderedSet for output stability

## Class: PackedNode

**Description:** A Packed Node represents a single derivation in a symbol node.

Parameters:
    rule: The rule associated with this node.
    parent: The parent of this node.
    left: The left child of this node. ``None`` if one does not exist.
    right: The right child of this node. ``None`` if one does not exist.
    priority: The priority of this node.

## Class: TokenNode

**Description:** A Token Node represents a matched terminal and is always a leaf node.

Parameters:
    token: The Token associated with this node.
    term: The TerminalDef matched by the token.
    priority: The priority of this node.

## Class: ForestVisitor

**Description:** An abstract base class for building forest visitors.

This class performs a controllable depth-first walk of an SPPF.
The visitor will not enter cycles and will backtrack if one is encountered.
Subclasses are notified of cycles through the ``on_cycle`` method.

Behavior for visit events is defined by overriding the
``visit*node*`` functions.

The walk is controlled by the return values of the ``visit*node_in``
methods. Returning a node(s) will schedule them to be visited. The visitor
will begin to backtrack if no nodes are returned.

Parameters:
    single_visit: If ``True``, non-Token nodes will only be visited once.

## Class: ForestTransformer

**Description:** The base class for a bottom-up forest transformation. Most users will
want to use ``TreeForestTransformer`` instead as it has a friendlier
interface and covers most use cases.

Transformations are applied via inheritance and overriding of the
``transform*node`` methods.

``transform_token_node`` receives a ``Token`` as an argument.
All other methods receive the node that is being transformed and
a list of the results of the transformations of that node's children.
The return value of these methods are the resulting transformations.

If ``Discard`` is raised in a node's transformation, no data from that node
will be passed to its parent's transformation.

## Class: ForestSumVisitor

**Description:** A visitor for prioritizing ambiguous parts of the Forest.

This visitor is used when support for explicit priorities on
rules is requested (whether normal, or invert). It walks the
forest (or subsets thereof) and cascades properties upwards
from the leaves.

It would be ideal to do this during parsing, however this would
require processing each Earley item multiple times. That's
a big performance drawback; so running a forest walk is the
lesser of two evils: there can be significantly more Earley
items created during parsing than there are SPPF nodes in the
final tree.

## Class: PackedData

**Description:** Used in transformationss of packed nodes to distinguish the data
that comes from the left child and the right child.

## Class: ForestToParseTree

**Description:** Used by the earley parser when ambiguity equals 'resolve' or
'explicit'. Transforms an SPPF into an (ambiguous) parse tree.

Parameters:
    tree_class: The tree class to use for construction
    callbacks: A dictionary of rules to functions that output a tree
    prioritizer: A ``ForestVisitor`` that manipulates the priorities of ForestNodes
    resolve_ambiguity: If True, ambiguities will be resolved based on
                    priorities. Otherwise, `_ambig` nodes will be in the resulting tree.
    use_cache: If True, the results of packed node transformations will be cached.

### Function: handles_ambiguity(func)

**Description:** Decorator for methods of subclasses of ``TreeForestTransformer``.
Denotes that the method should receive a list of transformed derivations.

## Class: TreeForestTransformer

**Description:** A ``ForestTransformer`` with a tree ``Transformer``-like interface.
By default, it will construct a tree.

Methods provided via inheritance are called based on the rule/symbol
names of nodes in the forest.

Methods that act on rules will receive a list of the results of the
transformations of the rule's children. By default, trees and tokens.

Methods that act on tokens will receive a token.

Alternatively, methods that act on rules may be annotated with
``handles_ambiguity``. In this case, the function will receive a list
of all the transformations of all the derivations of the rule.
By default, a list of trees where each tree.data is equal to the
rule name or one of its aliases.

Non-tree transformations are made possible by override of
``__default__``, ``__default_token__``, and ``__default_ambig__``.

Note:
    Tree shaping features such as inlined rules and token filtering are
    not built into the transformation. Positions are also not propagated.

Parameters:
    tree_class: The tree class to use for construction
    prioritizer: A ``ForestVisitor`` that manipulates the priorities of nodes in the SPPF.
    resolve_ambiguity: If True, ambiguities will be resolved based on priorities.
    use_cache (bool): If True, caches the results of some transformations,
                      potentially improving performance when ``resolve_ambiguity==False``.
                      Only use if you know what you are doing: i.e. All transformation
                      functions are pure and referentially transparent.

## Class: ForestToPyDotVisitor

**Description:** A Forest visitor which writes the SPPF to a PNG.

The SPPF can get really large, really quickly because
of the amount of meta-data it stores, so this is probably
only useful for trivial trees and learning how the SPPF
is structured.

### Function: __init__(self, s, start, end)

### Function: add_family(self, lr0, rule, start, left, right)

### Function: add_path(self, transitive, node)

### Function: load_paths(self)

### Function: is_ambiguous(self)

**Description:** Returns True if this node is ambiguous.

### Function: children(self)

**Description:** Returns a list of this node's children sorted from greatest to
least priority.

### Function: __iter__(self)

### Function: __repr__(self)

### Function: __init__(self, parent, s, rule, start, left, right)

### Function: is_empty(self)

### Function: sort_key(self)

**Description:** Used to sort PackedNode children of SymbolNodes.
A SymbolNode has multiple PackedNodes if it matched
ambiguously. Hence, we use the sort order to identify
the order in which ambiguous children should be considered.

### Function: children(self)

**Description:** Returns a list of this node's children.

### Function: __iter__(self)

### Function: __eq__(self, other)

### Function: __hash__(self)

### Function: __repr__(self)

### Function: __init__(self, token, term, priority)

### Function: __eq__(self, other)

### Function: __hash__(self)

### Function: __repr__(self)

### Function: __init__(self, single_visit)

### Function: visit_token_node(self, node)

**Description:** Called when a ``Token`` is visited. ``Token`` nodes are always leaves.

### Function: visit_symbol_node_in(self, node)

**Description:** Called when a symbol node is visited. Nodes that are returned
will be scheduled to be visited. If ``visit_intermediate_node_in``
is not implemented, this function will be called for intermediate
nodes as well.

### Function: visit_symbol_node_out(self, node)

**Description:** Called after all nodes returned from a corresponding ``visit_symbol_node_in``
call have been visited. If ``visit_intermediate_node_out``
is not implemented, this function will be called for intermediate
nodes as well.

### Function: visit_packed_node_in(self, node)

**Description:** Called when a packed node is visited. Nodes that are returned
will be scheduled to be visited. 

### Function: visit_packed_node_out(self, node)

**Description:** Called after all nodes returned from a corresponding ``visit_packed_node_in``
call have been visited.

### Function: on_cycle(self, node, path)

**Description:** Called when a cycle is encountered.

Parameters:
    node: The node that causes a cycle.
    path: The list of nodes being visited: nodes that have been
        entered but not exited. The first element is the root in a forest
        visit, and the last element is the node visited most recently.
        ``path`` should be treated as read-only.

### Function: get_cycle_in_path(self, node, path)

**Description:** A utility function for use in ``on_cycle`` to obtain a slice of
``path`` that only contains the nodes that make up the cycle.

### Function: visit(self, root)

### Function: __init__(self)

### Function: transform(self, root)

**Description:** Perform a transformation on an SPPF.

### Function: transform_symbol_node(self, node, data)

**Description:** Transform a symbol node.

### Function: transform_intermediate_node(self, node, data)

**Description:** Transform an intermediate node.

### Function: transform_packed_node(self, node, data)

**Description:** Transform a packed node.

### Function: transform_token_node(self, node)

**Description:** Transform a ``Token``.

### Function: visit_symbol_node_in(self, node)

### Function: visit_packed_node_in(self, node)

### Function: visit_token_node(self, node)

### Function: _visit_node_out_helper(self, node, method)

### Function: visit_symbol_node_out(self, node)

### Function: visit_intermediate_node_out(self, node)

### Function: visit_packed_node_out(self, node)

### Function: __init__(self)

### Function: visit_packed_node_in(self, node)

### Function: visit_symbol_node_in(self, node)

### Function: visit_packed_node_out(self, node)

### Function: visit_symbol_node_out(self, node)

## Class: _NoData

### Function: __init__(self, node, data)

### Function: __init__(self, tree_class, callbacks, prioritizer, resolve_ambiguity, use_cache)

### Function: visit(self, root)

### Function: on_cycle(self, node, path)

### Function: _check_cycle(self, node)

### Function: _collapse_ambig(self, children)

### Function: _call_rule_func(self, node, data)

### Function: _call_ambig_func(self, node, data)

### Function: transform_symbol_node(self, node, data)

### Function: transform_intermediate_node(self, node, data)

### Function: transform_packed_node(self, node, data)

### Function: visit_symbol_node_in(self, node)

### Function: visit_packed_node_in(self, node)

### Function: visit_packed_node_out(self, node)

### Function: __init__(self, tree_class, prioritizer, resolve_ambiguity, use_cache)

### Function: __default__(self, name, data)

**Description:** Default operation on tree (for override).

Returns a tree with name with data as children.

### Function: __default_ambig__(self, name, data)

**Description:** Default operation on ambiguous rule (for override).

Wraps data in an '_ambig_' node if it contains more than
one element.

### Function: __default_token__(self, node)

**Description:** Default operation on ``Token`` (for override).

Returns ``node``.

### Function: transform_token_node(self, node)

### Function: _call_rule_func(self, node, data)

### Function: _call_ambig_func(self, node, data)

### Function: __init__(self, rankdir)

### Function: visit(self, root, filename)

### Function: visit_token_node(self, node)

### Function: visit_packed_node_in(self, node)

### Function: visit_packed_node_out(self, node)

### Function: visit_symbol_node_in(self, node)

### Function: visit_symbol_node_out(self, node)
