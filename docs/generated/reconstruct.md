## AI Summary

A file named reconstruct.py.


### Function: is_iter_empty(i)

## Class: WriteTokensTransformer

**Description:** Inserts discarded tokens into their correct place, according to the rules of grammar

## Class: Reconstructor

**Description:** A Reconstructor that will, given a full parse Tree, generate source code.

Note:
    The reconstructor cannot generate values from regexps. If you need to produce discarded
    regexes, such as newlines, use `term_subs` and provide default values for them.

Parameters:
    parser: a Lark instance
    term_subs: a dictionary of [Terminal name as str] to [output text as str]

### Function: __init__(self, tokens, term_subs)

### Function: __default__(self, data, children, meta)

### Function: __init__(self, parser, term_subs)

### Function: _reconstruct(self, tree)

### Function: reconstruct(self, tree, postproc, insert_spaces)
