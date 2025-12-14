## AI Summary

A file named grammar_analysis.py.


## Class: RulePtr

## Class: LR0ItemSet

### Function: update_set(set1, set2)

### Function: calculate_sets(rules)

**Description:** Calculate FOLLOW sets.

Adapted from: http://lara.epfl.ch/w/cc09:algorithm_for_first_and_follow_sets

## Class: GrammarAnalyzer

### Function: __init__(self, rule, index)

### Function: __repr__(self)

### Function: next(self)

### Function: advance(self, sym)

### Function: is_satisfied(self)

### Function: __eq__(self, other)

### Function: __hash__(self)

### Function: __init__(self, kernel, closure)

### Function: __repr__(self)

### Function: __init__(self, parser_conf, debug, strict)

### Function: expand_rule(self, source_rule, rules_by_origin)

**Description:** Returns all init_ptrs accessible by rule (recursive)

### Function: _expand_rule(rule)
