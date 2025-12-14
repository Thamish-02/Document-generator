## AI Summary

A file named otTables.py.


## Class: VarComponentFlags

### Function: _read_uint32var(data, i)

**Description:** Read a variable-length number from data starting at index i.

Return the number and the next index.

### Function: _write_uint32var(v)

**Description:** Write a variable-length number.

Return the data.

## Class: VarComponent

## Class: VarCompositeGlyph

## Class: AATStateTable

## Class: AATState

## Class: AATAction

## Class: RearrangementMorphAction

## Class: ContextualMorphAction

## Class: LigAction

## Class: LigatureMorphAction

## Class: InsertionMorphAction

## Class: FeatureParams

## Class: FeatureParamsSize

## Class: FeatureParamsStylisticSet

## Class: FeatureParamsCharacterVariants

## Class: Coverage

## Class: DeltaSetIndexMap

## Class: VarIdxMap

## Class: VarRegionList

## Class: SingleSubst

## Class: MultipleSubst

## Class: ClassDef

## Class: AlternateSubst

## Class: LigatureSubst

## Class: COLR

## Class: LookupList

## Class: BaseGlyphRecordArray

## Class: BaseGlyphList

## Class: ClipBoxFormat

## Class: ClipBox

## Class: ClipList

## Class: ExtendMode

## Class: CompositeMode

## Class: PaintFormat

## Class: Paint

### Function: fixLookupOverFlows(ttf, overflowRecord)

**Description:** Either the offset from the LookupList to a lookup overflowed, or
an offset from a lookup to a subtable overflowed.

The table layout is::

  GPSO/GUSB
          Script List
          Feature List
          LookUpList
                  Lookup[0] and contents
                          SubTable offset list
                                  SubTable[0] and contents
                                  ...
                                  SubTable[n] and contents
                  ...
                  Lookup[n] and contents
                          SubTable offset list
                                  SubTable[0] and contents
                                  ...
                                  SubTable[n] and contents

If the offset to a lookup overflowed (SubTableIndex is None)
        we must promote the *previous* lookup to an Extension type.

If the offset from a lookup to subtable overflowed, then we must promote it
        to an Extension Lookup type.

### Function: splitMultipleSubst(oldSubTable, newSubTable, overflowRecord)

### Function: splitAlternateSubst(oldSubTable, newSubTable, overflowRecord)

### Function: splitLigatureSubst(oldSubTable, newSubTable, overflowRecord)

### Function: splitPairPos(oldSubTable, newSubTable, overflowRecord)

### Function: splitMarkBasePos(oldSubTable, newSubTable, overflowRecord)

### Function: fixSubTableOverFlows(ttf, overflowRecord)

**Description:** An offset has overflowed within a sub-table. We need to divide this subtable into smaller parts.

### Function: _buildClasses()

### Function: _getGlyphsFromCoverageTable(coverage)

### Function: __init__(self)

### Function: populateDefaults(self, propagator)

### Function: decompile(self, data, font, localState)

### Function: compile(self, font)

### Function: toXML(self, writer, ttFont, attrs)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: applyTransformDeltas(self, deltas)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __init__(self, components)

### Function: decompile(self, data, font, localState)

### Function: compile(self, font)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: fromXML(self, name, attrs, content, font)

### Function: __init__(self)

### Function: __init__(self)

### Function: compileActions(font, states)

### Function: _writeFlagsToXML(self, xmlWriter)

### Function: _setFlag(self, flag)

### Function: __init__(self)

### Function: compile(self, writer, font, actionIndex)

### Function: decompile(self, reader, font, actionReader)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: fromXML(self, name, attrs, content, font)

### Function: __init__(self)

### Function: compile(self, writer, font, actionIndex)

### Function: decompile(self, reader, font, actionReader)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: fromXML(self, name, attrs, content, font)

### Function: __init__(self)

### Function: __init__(self)

### Function: compile(self, writer, font, actionIndex)

### Function: decompile(self, reader, font, actionReader)

### Function: compileActions(font, states)

### Function: compileLigActions(self)

### Function: _decompileLigActions(self, actionReader, actionIndex)

### Function: fromXML(self, name, attrs, content, font)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: __init__(self)

### Function: compile(self, writer, font, actionIndex)

### Function: decompile(self, reader, font, actionReader)

### Function: _decompileInsertionAction(self, actionReader, font, index, count)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: fromXML(self, name, attrs, content, font)

### Function: compileActions(font, states)

### Function: compile(self, writer, font)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: populateDefaults(self, propagator)

### Function: postRead(self, rawTable, font)

### Function: preWrite(self, font)

### Function: toXML2(self, xmlWriter, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: populateDefaults(self, propagator)

### Function: postRead(self, rawTable, font)

### Function: getEntryFormat(mapping)

### Function: preWrite(self, font)

### Function: toXML2(self, xmlWriter, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: __getitem__(self, i)

### Function: populateDefaults(self, propagator)

### Function: postRead(self, rawTable, font)

### Function: preWrite(self, font)

### Function: toXML2(self, xmlWriter, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: __getitem__(self, glyphName)

### Function: preWrite(self, font)

### Function: populateDefaults(self, propagator)

### Function: postRead(self, rawTable, font)

### Function: preWrite(self, font)

### Function: toXML2(self, xmlWriter, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: populateDefaults(self, propagator)

### Function: postRead(self, rawTable, font)

### Function: preWrite(self, font)

### Function: toXML2(self, xmlWriter, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: makeSequence_(g)

### Function: populateDefaults(self, propagator)

### Function: postRead(self, rawTable, font)

### Function: _getClassRanges(self, font)

### Function: preWrite(self, font)

### Function: toXML2(self, xmlWriter, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: populateDefaults(self, propagator)

### Function: postRead(self, rawTable, font)

### Function: preWrite(self, font)

### Function: toXML2(self, xmlWriter, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: populateDefaults(self, propagator)

### Function: postRead(self, rawTable, font)

### Function: _getLigatureSortKey(components)

### Function: preWrite(self, font)

### Function: toXML2(self, xmlWriter, font)

### Function: fromXML(self, name, attrs, content, font)

### Function: decompile(self, reader, font)

### Function: preWrite(self, font)

### Function: computeClipBoxes(self, glyphSet, quantization)

### Function: table(self)

### Function: toXML2(self, xmlWriter, font)

### Function: preWrite(self, font)

### Function: preWrite(self, font)

### Function: is_variable(self)

### Function: as_variable(self)

### Function: as_tuple(self)

### Function: __repr__(self)

### Function: populateDefaults(self, propagator)

### Function: postRead(self, rawTable, font)

### Function: groups(self)

### Function: preWrite(self, font)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: fromXML(self, name, attrs, content, font)

### Function: is_variable(self)

### Function: as_variable(self)

### Function: getFormatName(self)

### Function: toXML(self, xmlWriter, font, attrs, name)

### Function: iterPaintSubTables(self, colr)

### Function: getChildren(self, colr)

### Function: traverse(self, colr, callback)

**Description:** Depth-first traversal of graph rooted at self, callback on each node.

### Function: getTransform(self)

### Function: computeClipBox(self, colr, glyphSet, quantization)

### Function: read_transform_component(values)

### Function: write_transform_component(value, values)

### Function: write(name, value, attrs)

### Function: read_transform_component_delta(values)
