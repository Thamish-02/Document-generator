## AI Summary

A file named ast_mod.py.


## Class: Mangler

**Description:** Mangle given names in and ast tree to make sure they do not conflict with
user code.

## Class: ReplaceCodeTransformer

### Function: log(self)

### Function: __init__(self, predicate)

### Function: visit_Name(self, node)

### Function: visit_FunctionDef(self, node)

### Function: visit_ImportFrom(self, node)

### Function: visit_Import(self, node)

### Function: _visit_Import_and_ImportFrom(self, node)

### Function: __init__(self, template, mapping, mangling_predicate)

### Function: from_string(cls, template, mapping, mangling_predicate)

### Function: visit_Module(self, code)

### Function: visit_Expr(self, expr)
