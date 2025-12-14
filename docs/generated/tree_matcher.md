## AI Summary

A file named tree_matcher.py.


### Function: is_discarded_terminal(t)

## Class: _MakeTreeMatch

### Function: _best_from_group(seq, group_key, cmp_key)

### Function: _best_rules_from_group(rules)

### Function: _match(term, token)

### Function: make_recons_rule(origin, expansion, old_expansion)

### Function: make_recons_rule_to_term(origin, term)

### Function: parse_rulename(s)

**Description:** Parse rule names that may contain a template syntax (like rule{a, b, ...})

## Class: ChildrenLexer

## Class: TreeMatcher

**Description:** Match the elements of a tree node, based on an ontology
provided by a Lark grammar.

Supports templates and inlined rules (`rule{a, b,..}` and `_rule`)

Initialize with an instance of Lark.

### Function: __init__(self, name, expansion)

### Function: __call__(self, args)

### Function: __init__(self, children)

### Function: lex(self, parser_state)

### Function: __init__(self, parser)

### Function: _build_recons_rules(self, rules)

**Description:** Convert tree-parsing/construction rules to tree-matching rules

### Function: match_tree(self, tree, rulename)

**Description:** Match the elements of `tree` to the symbols of rule `rulename`.

Parameters:
    tree (Tree): the tree node to match
    rulename (str): The expected full rule name (including template args)

Returns:
    Tree: an unreduced tree that matches `rulename`

Raises:
    UnexpectedToken: If no match was found.

Note:
    It's the callers' responsibility to match the tree recursively.
