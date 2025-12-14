## AI Summary

A file named psCharStrings.py.


### Function: read_operator(self, b0, data, index)

### Function: read_byte(self, b0, data, index)

### Function: read_smallInt1(self, b0, data, index)

### Function: read_smallInt2(self, b0, data, index)

### Function: read_shortInt(self, b0, data, index)

### Function: read_longInt(self, b0, data, index)

### Function: read_fixed1616(self, b0, data, index)

### Function: read_reserved(self, b0, data, index)

### Function: read_realNumber(self, b0, data, index)

### Function: buildOperatorDict(operatorList)

### Function: getIntEncoder(format)

### Function: encodeFixed(f, pack)

**Description:** For T2 only

### Function: encodeFloat(f)

## Class: CharStringCompileError

## Class: SimpleT2Decompiler

## Class: T2StackUseExtractor

## Class: T2WidthExtractor

## Class: T2OutlineExtractor

## Class: T1OutlineExtractor

## Class: T2CharString

## Class: T1CharString

## Class: DictDecompiler

### Function: calcSubrBias(subrs)

### Function: encodeInt(value, fourByteOp, bytechr, pack, unpack, twoByteOp)

### Function: __init__(self, localSubrs, globalSubrs, private, blender)

### Function: reset(self)

### Function: execute(self, charString)

### Function: pop(self)

### Function: popall(self)

### Function: push(self, value)

### Function: op_return(self, index)

### Function: op_endchar(self, index)

### Function: op_ignore(self, index)

### Function: op_callsubr(self, index)

### Function: op_callgsubr(self, index)

### Function: op_hstem(self, index)

### Function: op_vstem(self, index)

### Function: op_hstemhm(self, index)

### Function: op_vstemhm(self, index)

### Function: op_hintmask(self, index)

### Function: countHints(self)

### Function: op_and(self, index)

### Function: op_or(self, index)

### Function: op_not(self, index)

### Function: op_store(self, index)

### Function: op_abs(self, index)

### Function: op_add(self, index)

### Function: op_sub(self, index)

### Function: op_div(self, index)

### Function: op_load(self, index)

### Function: op_neg(self, index)

### Function: op_eq(self, index)

### Function: op_drop(self, index)

### Function: op_put(self, index)

### Function: op_get(self, index)

### Function: op_ifelse(self, index)

### Function: op_random(self, index)

### Function: op_mul(self, index)

### Function: op_sqrt(self, index)

### Function: op_dup(self, index)

### Function: op_exch(self, index)

### Function: op_index(self, index)

### Function: op_roll(self, index)

### Function: op_blend(self, index)

### Function: op_vsindex(self, index)

### Function: execute(self, charString)

### Function: __init__(self, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private, blender)

### Function: reset(self)

### Function: popallWidth(self, evenOdd)

### Function: countHints(self)

### Function: op_rmoveto(self, index)

### Function: op_hmoveto(self, index)

### Function: op_vmoveto(self, index)

### Function: op_endchar(self, index)

### Function: __init__(self, pen, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private, blender)

### Function: reset(self)

### Function: execute(self, charString)

### Function: _nextPoint(self, point)

### Function: rMoveTo(self, point)

### Function: rLineTo(self, point)

### Function: rCurveTo(self, pt1, pt2, pt3)

### Function: closePath(self)

### Function: endPath(self)

### Function: op_rmoveto(self, index)

### Function: op_hmoveto(self, index)

### Function: op_vmoveto(self, index)

### Function: op_endchar(self, index)

### Function: op_rlineto(self, index)

### Function: op_hlineto(self, index)

### Function: op_vlineto(self, index)

### Function: op_rrcurveto(self, index)

**Description:** {dxa dya dxb dyb dxc dyc}+ rrcurveto

### Function: op_rcurveline(self, index)

**Description:** {dxa dya dxb dyb dxc dyc}+ dxd dyd rcurveline

### Function: op_rlinecurve(self, index)

**Description:** {dxa dya}+ dxb dyb dxc dyc dxd dyd rlinecurve

### Function: op_vvcurveto(self, index)

**Description:** dx1? {dya dxb dyb dyc}+ vvcurveto

### Function: op_hhcurveto(self, index)

**Description:** dy1? {dxa dxb dyb dxc}+ hhcurveto

### Function: op_vhcurveto(self, index)

**Description:** dy1 dx2 dy2 dx3 {dxa dxb dyb dyc dyd dxe dye dxf}* dyf? vhcurveto (30)
{dya dxb dyb dxc dxd dxe dye dyf}+ dxf? vhcurveto

### Function: op_hvcurveto(self, index)

**Description:** dx1 dx2 dy2 dy3 {dya dxb dyb dxc dxd dxe dye dyf}* dxf?
{dxa dxb dyb dyc dyd dxe dye dxf}+ dyf?

### Function: op_hflex(self, index)

### Function: op_flex(self, index)

### Function: op_hflex1(self, index)

### Function: op_flex1(self, index)

### Function: op_and(self, index)

### Function: op_or(self, index)

### Function: op_not(self, index)

### Function: op_store(self, index)

### Function: op_abs(self, index)

### Function: op_add(self, index)

### Function: op_sub(self, index)

### Function: op_div(self, index)

### Function: op_load(self, index)

### Function: op_neg(self, index)

### Function: op_eq(self, index)

### Function: op_drop(self, index)

### Function: op_put(self, index)

### Function: op_get(self, index)

### Function: op_ifelse(self, index)

### Function: op_random(self, index)

### Function: op_mul(self, index)

### Function: op_sqrt(self, index)

### Function: op_dup(self, index)

### Function: op_exch(self, index)

### Function: op_index(self, index)

### Function: op_roll(self, index)

### Function: alternatingLineto(self, isHorizontal)

### Function: vcurveto(self, args)

### Function: hcurveto(self, args)

### Function: __init__(self, pen, subrs)

### Function: reset(self)

### Function: endPath(self)

### Function: popallWidth(self, evenOdd)

### Function: exch(self)

### Function: op_rmoveto(self, index)

### Function: op_hmoveto(self, index)

### Function: op_vmoveto(self, index)

### Function: op_closepath(self, index)

### Function: op_setcurrentpoint(self, index)

### Function: op_endchar(self, index)

### Function: op_hsbw(self, index)

### Function: op_sbw(self, index)

### Function: op_callsubr(self, index)

### Function: op_callothersubr(self, index)

### Function: op_pop(self, index)

### Function: doFlex(self)

### Function: op_dotsection(self, index)

### Function: op_hstem3(self, index)

### Function: op_seac(self, index)

**Description:** asb adx ady bchar achar seac

### Function: op_vstem3(self, index)

### Function: __init__(self, bytecode, program, private, globalSubrs)

### Function: getNumRegions(self, vsindex)

### Function: __repr__(self)

### Function: getIntEncoder(self)

### Function: getFixedEncoder(self)

### Function: decompile(self)

### Function: draw(self, pen, blender)

### Function: calcBounds(self, glyphSet)

### Function: compile(self, isCFF2)

### Function: needsDecompilation(self)

### Function: setProgram(self, program)

### Function: setBytecode(self, bytecode)

### Function: getToken(self, index, len, byteord, isinstance)

### Function: getBytes(self, index, nBytes)

### Function: handle_operator(self, operator)

### Function: toXML(self, xmlWriter, ttFont)

### Function: fromXML(self, name, attrs, content)

### Function: __init__(self, bytecode, program, subrs)

### Function: getIntEncoder(self)

### Function: getFixedEncoder(self)

### Function: decompile(self)

### Function: draw(self, pen)

### Function: __init__(self, strings, parent)

### Function: getDict(self)

### Function: decompile(self, data)

### Function: pop(self)

### Function: popall(self)

### Function: handle_operator(self, operator)

### Function: arg_number(self, name)

### Function: arg_blend_number(self, name)

### Function: arg_SID(self, name)

### Function: arg_array(self, name)

### Function: arg_blendList(self, name)

**Description:** There may be non-blend args at the top of the stack. We first calculate
where the blend args start in the stack. These are the last
numMasters*numBlends) +1 args.
The blend args starts with numMasters relative coordinate values, the  BlueValues in the list from the default master font. This is followed by
numBlends list of values. Each of  value in one of these lists is the
Variable Font delta for the matching region.

We re-arrange this to be a list of numMaster entries. Each entry starts with the corresponding default font relative value, and is followed by
the delta values. We then convert the default values, the first item in each entry, to an absolute value.

### Function: arg_delta(self, name)

### Function: pushToStack(value)

### Function: encodeFixed(value)
