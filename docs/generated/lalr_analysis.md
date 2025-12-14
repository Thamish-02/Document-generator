## AI Summary

A file named lalr_analysis.py.


## Class: Action

## Class: ParseTableBase

## Class: ParseTable

**Description:** Parse-table whose key is State, i.e. set[RulePtr]

Slower than IntParseTable, but useful for debugging

## Class: IntParseTable

**Description:** Parse-table whose key is int. Best for performance.

### Function: digraph(X, R, G)

### Function: traverse(x, S, N, X, R, G, F)

## Class: LALR_Analyzer

### Function: __init__(self, name)

### Function: __str__(self)

### Function: __repr__(self)

### Function: __init__(self, states, start_states, end_states)

### Function: serialize(self, memo)

### Function: deserialize(cls, data, memo)

### Function: from_ParseTable(cls, parse_table)

### Function: __init__(self, parser_conf, debug, strict)

### Function: compute_lr0_states(self)

### Function: compute_reads_relations(self)

### Function: compute_includes_lookback(self)

### Function: compute_lookaheads(self)

### Function: compute_lalr1_states(self)

### Function: compute_lalr(self)

### Function: step(state)
