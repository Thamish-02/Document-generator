## AI Summary

A file named TupleVariation.py.


## Class: TupleVariation

### Function: decompileSharedTuples(axisTags, sharedTupleCount, data, offset)

### Function: compileSharedTuples(axisTags, variations, MAX_NUM_SHARED_COORDS)

### Function: compileTupleVariationStore(variations, pointCount, axisTags, sharedTupleIndices, useSharedPoints)

### Function: decompileTupleVariationStore(tableTag, axisTags, tupleVariationCount, pointCount, sharedTuples, data, pos, dataPos)

### Function: decompileTupleVariation_(pointCount, sharedTuples, sharedPoints, tableTag, axisTags, data, tupleData)

### Function: inferRegion_(peak)

**Description:** Infer start and end for a (non-intermediate) region

This helper function computes the applicability region for
variation tuples whose INTERMEDIATE_REGION flag is not set in the
TupleVariationHeader structure.  Variation tuples apply only to
certain regions of the variation space; outside that region, the
tuple has no effect.  To make the binary encoding more compact,
TupleVariationHeaders can omit the intermediateStartTuple and
intermediateEndTuple fields.

### Function: __init__(self, axes, coordinates)

### Function: __repr__(self)

### Function: __eq__(self, other)

### Function: getUsedPoints(self)

### Function: hasImpact(self)

**Description:** Returns True if this TupleVariation has any visible impact.

If the result is False, the TupleVariation can be omitted from the font
without making any visible difference.

### Function: toXML(self, writer, axisTags)

### Function: fromXML(self, name, attrs, _content)

### Function: compile(self, axisTags, sharedCoordIndices, pointData)

### Function: compileCoord(self, axisTags)

### Function: compileIntermediateCoord(self, axisTags)

### Function: decompileCoord_(axisTags, data, offset)

### Function: compilePoints(points)

### Function: decompilePoints_(numPoints, data, offset, tableTag)

**Description:** (numPoints, data, offset, tableTag) --> ([point1, point2, ...], newOffset)

### Function: compileDeltas(self, optimizeSize)

### Function: compileDeltaValues_(deltas, bytearr)

**Description:** [value1, value2, value3, ...] --> bytearray

Emits a sequence of runs. Each run starts with a
byte-sized header whose 6 least significant bits
(header & 0x3F) indicate how many values are encoded
in this run. The stored length is the actual length
minus one; run lengths are thus in the range [1..64].
If the header byte has its most significant bit (0x80)
set, all values in this run are zero, and no data
follows. Otherwise, the header byte is followed by
((header & 0x3F) + 1) signed values.  If (header &
0x40) is clear, the delta values are stored as signed
bytes; if (header & 0x40) is set, the delta values are
signed 16-bit integers.

### Function: encodeDeltaRunAsZeroes_(deltas, offset, bytearr)

### Function: encodeDeltaRunAsBytes_(deltas, offset, bytearr, optimizeSize)

### Function: encodeDeltaRunAsWords_(deltas, offset, bytearr, optimizeSize)

### Function: encodeDeltaRunAsLongs_(deltas, offset, bytearr, optimizeSize)

### Function: decompileDeltas_(numDeltas, data, offset)

**Description:** (numDeltas, data, offset) --> ([delta, delta, ...], newOffset)

### Function: getTupleSize_(flags, axisCount)

### Function: getCoordWidth(self)

**Description:** Return 2 if coordinates are (x, y) as in gvar, 1 if single values
as in cvar, or 0 if empty.

### Function: scaleDeltas(self, scalar)

### Function: roundDeltas(self)

### Function: calcInferredDeltas(self, origCoords, endPts)

### Function: optimize(self, origCoords, endPts, tolerance, isComposite)

### Function: __imul__(self, scalar)

### Function: __iadd__(self, other)

### Function: key(pn)
