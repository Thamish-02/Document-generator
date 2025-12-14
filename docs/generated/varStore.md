## AI Summary

A file named varStore.py.


### Function: _getLocationKey(loc)

## Class: OnlineVarStoreBuilder

### Function: VarData_addItem(self, deltas)

### Function: VarRegion_get_support(self, fvar_axes)

### Function: VarStore___bool__(self)

## Class: VarStoreInstancer

### Function: VarStore_subset_varidxes(self, varIdxes, optimize, retainFirstMap, advIdxes)

### Function: VarStore_prune_regions(self)

**Description:** Remove unused VarRegions.

### Function: _visit(self, func)

**Description:** Recurse down from self, if type of an object is ot.Device,
call func() on it.  Works on otData-style classes.

### Function: _Device_recordVarIdx(self, s)

**Description:** Add VarIdx in this Device table (if any) to the set s.

### Function: Object_collect_device_varidxes(self, varidxes)

### Function: _Device_mapVarIdx(self, mapping, done)

**Description:** Map VarIdx in this Device table (if any) through mapping.

### Function: Object_remap_device_varidxes(self, varidxes_map)

## Class: _Encoding

## Class: _EncodingDict

### Function: VarStore_optimize(self, use_NO_VARIATION_INDEX, quantization)

**Description:** Optimize storage. Returns mapping from old VarIdxes to new ones.

### Function: main(args)

**Description:** Optimize a font's GDEF variation store

### Function: __init__(self, axisTags)

### Function: setModel(self, model)

### Function: setSupports(self, supports)

### Function: finish(self, optimize)

### Function: _add_VarData(self, num_items)

### Function: storeMasters(self, master_values)

### Function: storeMastersMany(self, master_values_list)

### Function: storeDeltas(self, deltas)

### Function: storeDeltasMany(self, deltas_list)

### Function: __init__(self, varstore, fvar_axes, location)

### Function: setLocation(self, location)

### Function: _clearCaches(self)

### Function: _getScalar(self, regionIdx)

### Function: interpolateFromDeltasAndScalars(deltas, scalars)

### Function: __getitem__(self, varidx)

### Function: interpolateFromDeltas(self, varDataIndex, deltas)

### Function: __init__(self, chars)

### Function: append(self, row)

### Function: extend(self, lst)

### Function: width_sort_key(self)

### Function: _characteristic_overhead(columns)

**Description:** Returns overhead in bytes of encoding this characteristic
as a VarData.

### Function: _columns(chars)

### Function: gain_from_merging(self, other_encoding)

### Function: __missing__(self, chars)

### Function: add_row(self, row)

### Function: _row_characteristics(row)

**Description:** Returns encoding characteristics for a row.
