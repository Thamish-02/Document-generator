## AI Summary

A file named otConverters.py.


### Function: buildConverters(tableSpec, tableNamespace)

**Description:** Given a table spec from otData.py, build a converter object for each
field of the table. This is called for each table in otData.py, and
the results are assigned to the corresponding class in otTables.py.

## Class: BaseConverter

**Description:** Base class for converter objects. Apart from the constructor, this
is an abstract class.

## Class: SimpleValue

## Class: OptionalValue

## Class: IntValue

## Class: Long

## Class: ULong

## Class: Flags32

## Class: VarIndex

## Class: Short

## Class: UShort

## Class: Int8

## Class: UInt8

## Class: UInt24

## Class: ComputedInt

## Class: ComputedUInt8

## Class: ComputedUShort

## Class: ComputedULong

## Class: Tag

## Class: GlyphID

## Class: GlyphID32

## Class: NameID

## Class: STATFlags

## Class: FloatValue

## Class: DeciPoints

## Class: BaseFixedValue

## Class: Fixed

## Class: F2Dot14

## Class: Angle

## Class: BiasedAngle

## Class: Version

## Class: Char64

**Description:** An ASCII string with up to 64 characters.

Unused character positions are filled with 0x00 bytes.
Used in Apple AAT fonts in the `gcid` table.

## Class: Struct

## Class: StructWithLength

## Class: Table

## Class: LTable

## Class: Table24

## Class: SubStruct

## Class: SubTable

## Class: ExtSubTable

## Class: FeatureParams

## Class: ValueFormat

## Class: ValueRecord

## Class: AATLookup

## Class: AATLookupWithDataOffset

## Class: MorxSubtableConverter

## Class: STXHeader

## Class: CIDGlyphMap

## Class: GlyphCIDMap

## Class: DeltaValue

## Class: VarIdxMapValue

## Class: VarDataValue

## Class: TupleValues

## Class: CFF2Index

## Class: LookupFlag

## Class: _UInt8Enum

## Class: ExtendMode

## Class: CompositeMode

### Function: __init__(self, name, repeat, aux, tableClass)

### Function: readArray(self, reader, font, tableDict, count)

**Description:** Read an array of values from the reader.

### Function: getRecordSize(self, reader)

### Function: read(self, reader, font, tableDict)

**Description:** Read a value from the reader.

### Function: writeArray(self, writer, font, tableDict, values)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

**Description:** Write a value to the writer.

### Function: xmlRead(self, attrs, content, font)

**Description:** Read a value from XML.

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

**Description:** Write a value to XML.

### Function: getVarIndexOffset(self)

**Description:** If description has `VarIndexBase + {offset}`, return the offset else None.

### Function: toString(value)

### Function: fromString(value)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: xmlRead(self, attrs, content, font)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: xmlRead(self, attrs, content, font)

### Function: fromString(value)

### Function: read(self, reader, font, tableDict)

### Function: readArray(self, reader, font, tableDict, count)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: writeArray(self, writer, font, tableDict, values)

### Function: read(self, reader, font, tableDict)

### Function: readArray(self, reader, font, tableDict, count)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: writeArray(self, writer, font, tableDict, values)

### Function: toString(value)

### Function: read(self, reader, font, tableDict)

### Function: readArray(self, reader, font, tableDict, count)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: writeArray(self, writer, font, tableDict, values)

### Function: read(self, reader, font, tableDict)

### Function: readArray(self, reader, font, tableDict, count)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: writeArray(self, writer, font, tableDict, values)

### Function: read(self, reader, font, tableDict)

### Function: readArray(self, reader, font, tableDict, count)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: writeArray(self, writer, font, tableDict, values)

### Function: read(self, reader, font, tableDict)

### Function: readArray(self, reader, font, tableDict, count)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: writeArray(self, writer, font, tableDict, values)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: readArray(self, reader, font, tableDict, count)

### Function: read(self, reader, font, tableDict)

### Function: writeArray(self, writer, font, tableDict, values)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: fromString(value)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: fromInt(cls, value)

### Function: toInt(cls, value)

### Function: fromString(cls, value)

### Function: toString(cls, value)

### Function: fromInt(cls, value)

### Function: toInt(cls, value)

### Function: fromString(cls, value)

### Function: toString(cls, value)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: fromString(value)

### Function: toString(value)

### Function: fromFloat(v)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: getRecordSize(self, reader)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: xmlRead(self, attrs, content, font)

### Function: __repr__(self)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: readOffset(self, reader)

### Function: writeNullOffset(self, writer)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: readOffset(self, reader)

### Function: writeNullOffset(self, writer)

### Function: readOffset(self, reader)

### Function: writeNullOffset(self, writer)

### Function: getConverter(self, tableType, lookupType)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: getConverter(self, tableType, lookupType)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: getConverter(self, featureTag)

### Function: __init__(self, name, repeat, aux, tableClass)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, format, repeatIndex)

### Function: getRecordSize(self, reader)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: xmlRead(self, attrs, content, font)

### Function: __init__(self, name, repeat, aux, tableClass)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: writeBinSearchHeader(writer, numUnits, unitSize)

### Function: buildFormat0(self, writer, font, values)

### Function: writeFormat0(self, writer, font, values)

### Function: buildFormat2(self, writer, font, values)

### Function: writeFormat2(self, writer, font, segments)

### Function: buildFormat6(self, writer, font, values)

### Function: writeFormat6(self, writer, font, values)

### Function: buildFormat8(self, writer, font, values)

### Function: writeFormat8(self, writer, font, values)

### Function: readFormat0(self, reader, font)

### Function: readFormat2(self, reader, font)

### Function: readFormat4(self, reader, font)

### Function: readFormat6(self, reader, font)

### Function: readFormat8(self, reader, font)

### Function: xmlRead(self, attrs, content, font)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: xmlRead(self, attrs, content, font)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: __init__(self, name, repeat, aux, tableClass)

### Function: _setTextDirectionFromCoverageFlags(self, flags, subtable)

### Function: read(self, reader, font, tableDict)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: xmlRead(self, attrs, content, font)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: __init__(self, name, repeat, aux, tableClass)

### Function: read(self, reader, font, tableDict)

### Function: _readTransition(self, reader, entryIndex, font, actionReader)

### Function: _readLigatures(self, reader, font)

### Function: _countPerGlyphLookups(self, table)

### Function: _readPerGlyphLookups(self, table, reader, font)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: _compileStates(self, font, states, glyphClassCount, actionIndex)

### Function: _compilePerGlyphLookups(self, table, font)

### Function: _compileLigComponents(self, table, font)

### Function: _compileLigatures(self, table, font)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: _xmlWriteLigatures(self, xmlWriter, font, value, name, attrs)

### Function: xmlRead(self, attrs, content, font)

### Function: _xmlReadState(self, attrs, content, font)

### Function: _xmlReadLigComponents(self, attrs, content, font)

### Function: _xmlReadLigatures(self, attrs, content, font)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: xmlRead(self, attrs, content, font)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: xmlRead(self, attrs, content, font)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: xmlRead(self, attrs, content, font)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, value, repeatIndex)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, values, repeatIndex)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: xmlRead(self, attrs, content, font)

### Function: read(self, data, font)

### Function: write(self, writer, font, tableDict, values, repeatIndex)

### Function: xmlRead(self, attrs, content, font)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: __init__(self, name, repeat, aux, tableClass)

### Function: read(self, reader, font, tableDict)

### Function: write(self, writer, font, tableDict, values, repeatIndex)

### Function: xmlRead(self, attrs, content, font)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: xmlWrite(self, xmlWriter, font, value, name, attrs)

### Function: read(self, reader, font, tableDict)

### Function: fromString(cls, value)

### Function: toString(cls, value)

### Function: getReadArray(reader, offSize)

### Function: get_read_item()

### Function: get_read_item()

### Function: read_item(i)

### Function: read_item(i)
