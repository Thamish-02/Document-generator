## AI Summary

A file named reorderGlyphs.py.


### Function: _sort_by_gid(get_glyph_id, glyphs, parallel_list)

### Function: _get_dotted_attr(value, dotted_attr)

## Class: ReorderRule

**Description:** A rule to reorder something in a font to match the fonts glyph order.

## Class: ReorderCoverage

**Description:** Reorder a Coverage table, and optionally a list that is sorted parallel to it.

## Class: ReorderList

**Description:** Reorder the items within a list to match the updated glyph order.

Useful when a list ordered by coverage itself contains something ordered by a gid.
For example, the PairSet table of https://docs.microsoft.com/en-us/typography/opentype/spec/gpos#lookup-type-2-pair-adjustment-positioning-subtable.

### Function: _bfs_base_table(root, root_accessor)

### Function: _traverse_ot_data(root, root_accessor, add_to_frontier_fn)

### Function: reorderGlyphs(font, new_glyph_order)

### Function: apply(self, font, value)

### Function: apply(self, font, value)

### Function: apply(self, font, value)
