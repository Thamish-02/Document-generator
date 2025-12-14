## AI Summary

A file named cyk.py.


### Function: match(t, s)

## Class: Rule

**Description:** Context-free grammar rule.

## Class: Grammar

**Description:** Context-free grammar.

## Class: RuleNode

**Description:** A node in the parse tree, which also contains the full rhs rule.

## Class: Parser

**Description:** Parser wrapper.

### Function: print_parse(node, indent)

### Function: _parse(s, g)

**Description:** Parses sentence 's' using CNF grammar 'g'.

## Class: CnfWrapper

**Description:** CNF wrapper for grammar.

Validates that the input grammar is CNF and provides helper data structures.

## Class: UnitSkipRule

**Description:** A rule that records NTs that were skipped during transformation.

### Function: build_unit_skiprule(unit_rule, target_rule)

### Function: get_any_nt_unit_rule(g)

**Description:** Returns a non-terminal unit rule from 'g', or None if there is none.

### Function: _remove_unit_rule(g, rule)

**Description:** Removes 'rule' from 'g' without changing the language produced by 'g'.

### Function: _split(rule)

**Description:** Splits a rule whose len(rhs) > 2 into shorter rules.

### Function: _term(g)

**Description:** Applies the TERM rule on 'g' (see top comment).

### Function: _bin(g)

**Description:** Applies the BIN rule to 'g' (see top comment).

### Function: _unit(g)

**Description:** Applies the UNIT rule to 'g' (see top comment).

### Function: to_cnf(g)

**Description:** Creates a CNF grammar from a general context-free grammar 'g'.

### Function: unroll_unit_skiprule(lhs, orig_rhs, skipped_rules, children, weight, alias)

### Function: revert_cnf(node)

**Description:** Reverts a parse tree (RuleNode) to its original non-CNF form (Node).

### Function: __init__(self, lhs, rhs, weight, alias)

### Function: __str__(self)

### Function: __repr__(self)

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __init__(self, rules)

### Function: __eq__(self, other)

### Function: __str__(self)

### Function: __repr__(self)

### Function: __init__(self, rule, children, weight)

### Function: __repr__(self)

### Function: __init__(self, rules)

### Function: _to_rule(self, lark_rule)

**Description:** Converts a lark rule, (lhs, rhs, callback, options), to a Rule.

### Function: parse(self, tokenized, start)

**Description:** Parses input, which is a list of tokens.

### Function: _to_tree(self, rule_node)

**Description:** Converts a RuleNode parse tree to a lark Tree.

### Function: __init__(self, grammar)

### Function: __eq__(self, other)

### Function: __repr__(self)

### Function: __init__(self, lhs, rhs, skipped_rules, weight, alias)

### Function: __eq__(self, other)
