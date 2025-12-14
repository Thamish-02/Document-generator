## AI Summary

A file named grammar.py.


## Class: Symbol

## Class: Terminal

## Class: NonTerminal

## Class: RuleOptions

## Class: Rule

**Description:** origin : a symbol
expansion : a list of symbols
order : index of this expansion amongst all rules of the same name

### Function: __init__(self, name)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __hash__(self)

### Function: __repr__(self)

### Function: renamed(self, f)

### Function: __init__(self, name, filter_out)

### Function: fullrepr(self)

### Function: renamed(self, f)

### Function: serialize(self, memo)

### Function: __init__(self, keep_all_tokens, expand1, priority, template_source, empty_indices)

### Function: __repr__(self)

### Function: __init__(self, origin, expansion, order, alias, options)

### Function: _deserialize(self)

### Function: __str__(self)

### Function: __repr__(self)

### Function: __hash__(self)

### Function: __eq__(self, other)
