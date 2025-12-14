## AI Summary

A file named merger.py.


## Class: Merger

## Class: AligningMerger

### Function: merge(merger, self, lst)

### Function: _SinglePosUpgradeToFormat2(self)

### Function: _merge_GlyphOrders(font, lst, values_lst, default)

**Description:** Takes font and list of glyph lists (must be sorted by glyph id), and returns
two things:
- Combined glyph list,
- If values_lst is None, return input glyph lists, but padded with None when a glyph
  was missing in a list.  Otherwise, return values_lst list-of-list, padded with None
  to match combined glyph lists.

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: _Lookup_SinglePos_get_effective_value(merger, subtables, glyph)

### Function: _Lookup_PairPos_get_effective_value_pair(merger, subtables, firstGlyph, secondGlyph)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: _PairPosFormat1_merge(self, lst, merger)

### Function: _ClassDef_invert(self, allGlyphs)

### Function: _ClassDef_merge_classify(lst, allGlyphses)

### Function: _PairPosFormat2_align_matrices(self, lst, font, transparent)

### Function: _PairPosFormat2_merge(self, lst, merger)

### Function: merge(merger, self, lst)

### Function: _MarkBasePosFormat1_merge(self, lst, merger, Mark, Base)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: _PairSet_flatten(lst, font)

### Function: _Lookup_PairPosFormat1_subtables_flatten(lst, font)

### Function: _Lookup_PairPosFormat2_subtables_flatten(lst, font)

### Function: _Lookup_PairPos_subtables_canonicalize(lst, font)

**Description:** Merge multiple Format1 subtables at the beginning of lst,
and merge multiple consecutive Format2 subtables that have the same
Class2 (ie. were split because of offset overflows).  Returns new list.

### Function: _Lookup_SinglePos_subtables_flatten(lst, font, min_inclusive_rec_format)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

## Class: InstancerMerger

**Description:** A merger that takes multiple master fonts, and instantiates
an instance.

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

## Class: MutatorMerger

**Description:** A merger that takes a variable font, and instantiates
an instance.  While there's no "merging" to be done per se,
the operation can benefit from many operations that the
aligning merger does.

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

## Class: VariationMerger

**Description:** A merger that takes multiple master fonts, and builds a
variable font.

### Function: buildVarDevTable(store_builder, master_values)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

## Class: COLRVariationMerger

**Description:** A specialized VariationMerger that takes multiple master fonts containing
COLRv1 tables, and builds a variable COLR font.

COLR tables are special in that variable subtables can be associated with
multiple delta-set indices (via VarIndexBase).
They also contain tables that must change their type (not simply the Format)
as they become variable (e.g. Affine2x3 -> VarAffine2x3) so this merger takes
care of that too.

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: _flatten_layers(root, colr)

### Function: _merge_PaintColrLayers(self, out, lst)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: merge(merger, self, lst)

### Function: __init__(self, font)

### Function: merger(celf, clazzes, attrs)

### Function: mergersFor(celf, thing, _default)

### Function: mergeObjects(self, out, lst, exclude)

### Function: mergeLists(self, out, lst)

### Function: mergeThings(self, out, lst)

### Function: mergeTables(self, font, master_ttfs, tableTags)

### Function: __init__(self, font, model, location)

### Function: __init__(self, font, instancer, deleteVariations)

### Function: __init__(self, model, axisTags, font)

### Function: setModel(self, model)

### Function: mergeThings(self, out, lst)

### Function: __init__(self, model, axisTags, font, allowLayerReuse)

### Function: mergeTables(self, font, master_ttfs, tableTags)

### Function: checkFormatEnum(self, out, lst, validate)

### Function: mergeSparseDict(self, out, lst)

### Function: mergeAttrs(self, out, lst, attrs)

### Function: storeMastersForAttr(self, out, lst, attr)

### Function: storeVariationIndices(self, varIdxes)

### Function: mergeVariableAttrs(self, out, lst, attrs)

### Function: convertSubTablesToVarType(cls, table)

### Function: expandPaintColrLayers(colr)

**Description:** Rebuild LayerList without PaintColrLayers reuse.

Each base paint graph is fully DFS-traversed (with exception of PaintColrGlyph
which are irrelevant for this); any layers referenced via PaintColrLayers are
collected into a new LayerList and duplicated when reuse is detected, to ensure
that all paints are distinct objects at the end of the process.
PaintColrLayers's FirstLayerIndex/NumLayers are updated so that no overlap
is left. Also, any consecutively nested PaintColrLayers are flattened.
The COLR table's LayerList is replaced with the new unique layers.
A side effect is also that any layer from the old LayerList which is not
referenced by any PaintColrLayers is dropped.

### Function: listToColrLayers(paint)

### Function: wrapper(method)
