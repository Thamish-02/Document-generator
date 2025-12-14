## AI Summary

A file named idtracking.py.


### Function: find_symbols(nodes, parent_symbols)

### Function: symbols_for_node(node, parent_symbols)

## Class: Symbols

## Class: RootVisitor

## Class: FrameSymbolVisitor

**Description:** A visitor for `Frame.inspect`.

### Function: __init__(self, parent, level)

### Function: analyze_node(self, node)

### Function: _define_ref(self, name, load)

### Function: find_load(self, target)

### Function: find_ref(self, name)

### Function: ref(self, name)

### Function: copy(self)

### Function: store(self, name)

### Function: declare_parameter(self, name)

### Function: load(self, name)

### Function: branch_update(self, branch_symbols)

### Function: dump_stores(self)

### Function: dump_param_targets(self)

### Function: __init__(self, symbols)

### Function: _simple_visit(self, node)

### Function: visit_AssignBlock(self, node)

### Function: visit_CallBlock(self, node)

### Function: visit_OverlayScope(self, node)

### Function: visit_For(self, node, for_branch)

### Function: visit_With(self, node)

### Function: generic_visit(self, node)

### Function: __init__(self, symbols)

### Function: visit_Name(self, node, store_as_param)

**Description:** All assignments to names go through this function.

### Function: visit_NSRef(self, node)

### Function: visit_If(self, node)

### Function: visit_Macro(self, node)

### Function: visit_Import(self, node)

### Function: visit_FromImport(self, node)

### Function: visit_Assign(self, node)

**Description:** Visit assignments in the correct order.

### Function: visit_For(self, node)

**Description:** Visiting stops at for blocks.  However the block sequence
is visited as part of the outer scope.

### Function: visit_CallBlock(self, node)

### Function: visit_FilterBlock(self, node)

### Function: visit_With(self, node)

### Function: visit_AssignBlock(self, node)

**Description:** Stop visiting at block assigns.

### Function: visit_Scope(self, node)

**Description:** Stop visiting at scopes.

### Function: visit_Block(self, node)

**Description:** Stop visiting at blocks.

### Function: visit_OverlayScope(self, node)

**Description:** Do not visit into overlay scopes.

### Function: inner_visit(nodes)
