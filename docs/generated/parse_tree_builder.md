## AI Summary

A file named parse_tree_builder.py.


## Class: ExpandSingleChild

## Class: PropagatePositions

### Function: make_propagate_positions(option)

## Class: ChildFilter

## Class: ChildFilterLALR

**Description:** Optimized childfilter for LALR (assumes no duplication in parse tree, so it's safe to change it)

## Class: ChildFilterLALR_NoPlaceholders

**Description:** Optimized childfilter for LALR (assumes no duplication in parse tree, so it's safe to change it)

### Function: _should_expand(sym)

### Function: maybe_create_child_filter(expansion, keep_all_tokens, ambiguous, _empty_indices)

## Class: AmbiguousExpander

**Description:** Deal with the case where we're expanding children ('_rule') into a parent but the children
are ambiguous. i.e. (parent->_ambig->_expand_this_rule). In this case, make the parent itself
ambiguous with as many copies as there are ambiguous children, and then copy the ambiguous children
into the right parents in the right places, essentially shifting the ambiguity up the tree.

### Function: maybe_create_ambiguous_expander(tree_class, expansion, keep_all_tokens)

## Class: AmbiguousIntermediateExpander

**Description:** Propagate ambiguous intermediate nodes and their derivations up to the
current rule.

In general, converts

rule
  _iambig
    _inter
      someChildren1
      ...
    _inter
      someChildren2
      ...
  someChildren3
  ...

to

_ambig
  rule
    someChildren1
    ...
    someChildren3
    ...
  rule
    someChildren2
    ...
    someChildren3
    ...
  rule
    childrenFromNestedIambigs
    ...
    someChildren3
    ...
  ...

propagating up any nested '_iambig' nodes along the way.

### Function: inplace_transformer(func)

### Function: apply_visit_wrapper(func, name, wrapper)

## Class: ParseTreeBuilder

### Function: __init__(self, node_builder)

### Function: __call__(self, children)

### Function: __init__(self, node_builder, node_filter)

### Function: __call__(self, children)

### Function: _pp_get_meta(self, children)

### Function: __init__(self, to_include, append_none, node_builder)

### Function: __call__(self, children)

### Function: __call__(self, children)

### Function: __init__(self, to_include, node_builder)

### Function: __call__(self, children)

### Function: __init__(self, to_expand, tree_class, node_builder)

### Function: __call__(self, children)

### Function: __init__(self, tree_class, node_builder)

### Function: __call__(self, children)

### Function: f(children)

### Function: f(children)

### Function: __init__(self, rules, tree_class, propagate_positions, ambiguous, maybe_placeholders)

### Function: _init_builders(self, rules)

### Function: create_callback(self, transformer)

### Function: _is_ambig_tree(t)

### Function: _is_iambig_tree(child)

### Function: _collapse_iambig(children)

**Description:** Recursively flatten the derivations of the parent of an '_iambig'
node. Returns a list of '_inter' nodes guaranteed not
to contain any nested '_iambig' nodes, or None if children does
not contain an '_iambig' node.

### Function: default_callback(data, children)
