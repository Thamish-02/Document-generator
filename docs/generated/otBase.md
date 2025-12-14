## AI Summary

A file named otBase.py.


## Class: OverflowErrorRecord

## Class: OTLOffsetOverflowError

## Class: RepackerState

## Class: BaseTTXConverter

**Description:** Generic base class for TTX table converters. It functions as an
adapter between the TTX (ttLib actually) table model and the model
we use for OpenType tables, which is necessarily subtly different.

## Class: OTTableReader

**Description:** Helper class to retrieve data from an OpenType table.

## Class: OffsetToWriter

## Class: OTTableWriter

**Description:** Helper class to gather and assemble data for OpenType tables.

## Class: CountReference

**Description:** A reference to a Count value, not a count of references.

### Function: packUInt8(value)

### Function: packUShort(value)

### Function: packULong(value)

### Function: packUInt24(value)

## Class: BaseTable

**Description:** Generic base class for all OpenType (sub)tables.

## Class: FormatSwitchingBaseTable

**Description:** Minor specialization of BaseTable, for tables that have multiple
formats, eg. CoverageFormat1 vs. CoverageFormat2.

## Class: UInt8FormatSwitchingBaseTable

### Function: getFormatSwitchingBaseTableClass(formatType)

### Function: getVariableAttrs(cls, fmt)

**Description:** Return sequence of variable table field names (can be empty).

Attributes are deemed "variable" when their otData.py's description contain
'VarIndexBase + {offset}', e.g. COLRv1 PaintVar* tables.

### Function: _buildDict()

## Class: ValueRecordFactory

**Description:** Given a format code, this object convert ValueRecords.

## Class: ValueRecord

### Function: __init__(self, overflowTuple)

### Function: __repr__(self)

### Function: __init__(self, overflowErrorRecord)

### Function: __str__(self)

### Function: decompile(self, data, font)

**Description:** Create an object from the binary data. Called automatically on access.

### Function: compile(self, font)

**Description:** Compiles the table into binary. Called automatically on save.

### Function: tryPackingHarfbuzz(self, writer, hb_first_error_logged)

### Function: tryPackingFontTools(self, writer)

### Function: tryResolveOverflow(self, font, e, lastOverflowRecord)

### Function: toXML(self, writer, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: ensureDecompiled(self, recurse)

### Function: __init__(self, data, localState, offset, tableTag)

### Function: advance(self, count)

### Function: seek(self, pos)

### Function: copy(self)

### Function: getSubReader(self, offset)

### Function: readValue(self, typecode, staticSize)

### Function: readArray(self, typecode, staticSize, count)

### Function: readInt8(self)

### Function: readInt8Array(self, count)

### Function: readShort(self)

### Function: readShortArray(self, count)

### Function: readLong(self)

### Function: readLongArray(self, count)

### Function: readUInt8(self)

### Function: readUInt8Array(self, count)

### Function: readUShort(self)

### Function: readUShortArray(self, count)

### Function: readULong(self)

### Function: readULongArray(self, count)

### Function: readUInt24(self)

### Function: readUInt24Array(self, count)

### Function: readTag(self)

### Function: readData(self, count)

### Function: __setitem__(self, name, value)

### Function: __getitem__(self, name)

### Function: __contains__(self, name)

### Function: __init__(self, subWriter, offsetSize)

### Function: __eq__(self, other)

### Function: __hash__(self)

### Function: __init__(self, localState, tableTag)

### Function: __setitem__(self, name, value)

### Function: __getitem__(self, name)

### Function: __delitem__(self, name)

### Function: getDataLength(self)

**Description:** Return the length of this table in bytes, without subtables.

### Function: getData(self)

**Description:** Assemble the data for this writer/table, without subtables.

### Function: getDataForHarfbuzz(self)

**Description:** Assemble the data for this writer/table with all offset field set to 0

### Function: __hash__(self)

### Function: __ne__(self, other)

### Function: __eq__(self, other)

### Function: _doneWriting(self, internedTables, shareExtension)

### Function: _gatherTables(self, tables, extTables, done)

### Function: _gatherGraphForHarfbuzz(self, tables, obj_list, done, objidx, virtual_edges)

### Function: getAllDataUsingHarfbuzz(self, tableTag)

**Description:** The Whole table is represented as a Graph.
Assemble graph data and call Harfbuzz repacker to pack the table.
Harfbuzz repacker is faster and retain as much sub-table sharing as possible, see also:
https://github.com/harfbuzz/harfbuzz/blob/main/docs/repacker.md
The input format for hb.repack() method is explained here:
https://github.com/harfbuzz/uharfbuzz/blob/main/src/uharfbuzz/_harfbuzz.pyx#L1149

### Function: getAllData(self, remove_duplicate)

**Description:** Assemble all data, including all subtables.

### Function: getSubWriter(self)

### Function: writeValue(self, typecode, value)

### Function: writeArray(self, typecode, values)

### Function: writeInt8(self, value)

### Function: writeInt8Array(self, values)

### Function: writeShort(self, value)

### Function: writeShortArray(self, values)

### Function: writeLong(self, value)

### Function: writeLongArray(self, values)

### Function: writeUInt8(self, value)

### Function: writeUInt8Array(self, values)

### Function: writeUShort(self, value)

### Function: writeUShortArray(self, values)

### Function: writeULong(self, value)

### Function: writeULongArray(self, values)

### Function: writeUInt24(self, value)

### Function: writeUInt24Array(self, values)

### Function: writeTag(self, tag)

### Function: writeSubTable(self, subWriter, offsetSize)

### Function: writeCountReference(self, table, name, size, value)

### Function: writeStruct(self, format, values)

### Function: writeData(self, data)

### Function: getOverflowErrorRecord(self, item)

### Function: __init__(self, table, name, size, value)

### Function: setValue(self, value)

### Function: getValue(self)

### Function: getCountData(self)

### Function: __getattr__(self, attr)

### Function: ensureDecompiled(self, recurse)

### Function: __getstate__(self)

### Function: getRecordSize(cls, reader)

### Function: getConverters(self)

### Function: getConverterByName(self, name)

### Function: populateDefaults(self, propagator)

### Function: decompile(self, reader, font)

### Function: compile(self, writer, font)

### Function: readFormat(self, reader)

### Function: writeFormat(self, writer)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: toXML2(self, xmlWriter, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: __ne__(self, other)

### Function: __eq__(self, other)

## Class: SubTableEntry

**Description:** See BaseTable.iterSubTables()

### Function: iterSubTables(self)

**Description:** Yield (name, value, index) namedtuples for all subtables of current table.

A sub-table is an instance of BaseTable (or subclass thereof) that is a child
of self, the current parent table.
The tuples also contain the attribute name (str) of the of parent table to get
a subtable, and optionally, for lists of subtables (i.e. attributes associated
with a converter that has a 'repeat'), an index into the list containing the
given subtable value.
This method can be useful to traverse trees of otTables.

### Function: getVariableAttrs(self)

### Function: getRecordSize(cls, reader)

### Function: getConverters(self)

### Function: getConverterByName(self, name)

### Function: readFormat(self, reader)

### Function: writeFormat(self, writer)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: getVariableAttrs(self)

### Function: readFormat(self, reader)

### Function: writeFormat(self, writer)

### Function: __init__(self, valueFormat)

### Function: __len__(self)

### Function: readValueRecord(self, reader, font)

### Function: writeValueRecord(self, writer, font, valueRecord)

### Function: __init__(self, valueFormat, src)

### Function: getFormat(self)

### Function: getEffectiveFormat(self)

### Function: toXML(self, xmlWriter, font, valueName, attrs)

### Function: fromXML(self, name, attrs, content, font)

### Function: __ne__(self, other)

### Function: __eq__(self, other)
