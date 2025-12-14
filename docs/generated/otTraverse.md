## AI Summary

A file named otTraverse.py.


## Class: SubTablePath

### Function: dfs_base_table(root, root_accessor, skip_root, predicate, iter_subtables_fn)

**Description:** Depth-first search tree of BaseTables.

Args:
    root (BaseTable): the root of the tree.
    root_accessor (Optional[str]): attribute name for the root table, if any (mostly
        useful for debugging).
    skip_root (Optional[bool]): if True, the root itself is not visited, only its
        children.
    predicate (Optional[Callable[[SubTablePath], bool]]): function to filter out
        paths. If True, the path is yielded and its subtables are added to the
        queue. If False, the path is skipped and its subtables are not traversed.
    iter_subtables_fn (Optional[Callable[[BaseTable], Iterable[BaseTable.SubTableEntry]]]):
        function to iterate over subtables of a table. If None, the default
        BaseTable.iterSubTables() is used.

Yields:
    SubTablePath: tuples of BaseTable.SubTableEntry(name, table, index) namedtuples
    for each of the nodes in the tree. The last entry in a path is the current
    subtable, whereas preceding ones refer to its parent tables all the way up to
    the root.

### Function: bfs_base_table(root, root_accessor, skip_root, predicate, iter_subtables_fn)

**Description:** Breadth-first search tree of BaseTables.

Args:
    root
        the root of the tree.
    root_accessor (Optional[str]): attribute name for the root table, if any (mostly
        useful for debugging).
    skip_root (Optional[bool]): if True, the root itself is not visited, only its
        children.
    predicate (Optional[Callable[[SubTablePath], bool]]): function to filter out
        paths. If True, the path is yielded and its subtables are added to the
        queue. If False, the path is skipped and its subtables are not traversed.
    iter_subtables_fn (Optional[Callable[[BaseTable], Iterable[BaseTable.SubTableEntry]]]):
        function to iterate over subtables of a table. If None, the default
        BaseTable.iterSubTables() is used.

Yields:
    SubTablePath: tuples of BaseTable.SubTableEntry(name, table, index) namedtuples
    for each of the nodes in the tree. The last entry in a path is the current
    subtable, whereas preceding ones refer to its parent tables all the way up to
    the root.

### Function: _traverse_ot_data(root, root_accessor, skip_root, predicate, add_to_frontier_fn, iter_subtables_fn)

### Function: __str__(self)

### Function: predicate(path)

### Function: iter_subtables_fn(table)
