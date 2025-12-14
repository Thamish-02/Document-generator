## AI Summary

A file named mark_tokens.py.


## Class: MarkTokens

**Description:** Helper that visits all nodes in the AST tree and assigns .first_token and .last_token attributes
to each of them. This is the heart of the token-marking logic.

### Function: __init__(self, code)

### Function: visit_tree(self, node)

### Function: _visit_before_children(self, node, parent_token)

### Function: _visit_after_children(self, node, parent_token, token)

### Function: _find_last_in_stmt(self, start_token)

### Function: _expand_to_matching_pairs(self, first_token, last_token, node)

**Description:** Scan tokens in [first_token, last_token] range that are between node's children, and for any
unmatched brackets, adjust first/last tokens to include the closing pair.

### Function: visit_default(self, node, first_token, last_token)

### Function: handle_comp(self, open_brace, node, first_token, last_token)

### Function: visit_comprehension(self, node, first_token, last_token)

### Function: visit_if(self, node, first_token, last_token)

### Function: handle_attr(self, node, first_token, last_token)

### Function: handle_def(self, node, first_token, last_token)

### Function: handle_following_brackets(self, node, last_token, opening_bracket)

### Function: visit_call(self, node, first_token, last_token)

### Function: visit_matchclass(self, node, first_token, last_token)

### Function: visit_subscript(self, node, first_token, last_token)

### Function: visit_slice(self, node, first_token, last_token)

### Function: handle_bare_tuple(self, node, first_token, last_token)

### Function: handle_tuple_nonempty(self, node, first_token, last_token)

### Function: visit_tuple(self, node, first_token, last_token)

### Function: _gobble_parens(self, first_token, last_token, include_all)

### Function: visit_str(self, node, first_token, last_token)

### Function: visit_joinedstr(self, node, first_token, last_token)

### Function: visit_bytes(self, node, first_token, last_token)

### Function: handle_str(self, first_token, last_token)

### Function: handle_num(self, node, value, first_token, last_token)

### Function: visit_num(self, node, first_token, last_token)

### Function: visit_const(self, node, first_token, last_token)

### Function: visit_keyword(self, node, first_token, last_token)

### Function: visit_starred(self, node, first_token, last_token)

### Function: visit_assignname(self, node, first_token, last_token)

### Function: handle_async(self, node, first_token, last_token)

### Function: visit_asyncfunctiondef(self, node, first_token, last_token)
