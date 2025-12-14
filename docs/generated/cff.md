## AI Summary

A file named cff.py.


### Function: addCFFVarStore(varFont, varModel, varDataList, masterSupports)

### Function: convertCFFtoCFF2(varFont)

### Function: conv_to_int(num)

### Function: get_private(regionFDArrays, fd_index, ri, fd_map)

### Function: merge_PrivateDicts(top_dicts, vsindex_dict, var_model, fd_map)

**Description:** I step through the FontDicts in the FDArray of the varfont TopDict.
For each varfont FontDict:

* step through each key in FontDict.Private.
* For each key, step through each relevant source font Private dict, and
  build a list of values to blend.

The 'relevant' source fonts are selected by first getting the right
submodel using ``vsindex_dict[vsindex]``. The indices of the
``subModel.locations`` are mapped to source font list indices by
assuming the latter order is the same as the order of the
``var_model.locations``. I can then get the index of each subModel
location in the list of ``var_model.locations``.

### Function: _cff_or_cff2(font)

### Function: getfd_map(varFont, fonts_list)

**Description:** Since a subset source font may have fewer FontDicts in their
FDArray than the default font, we have to match up the FontDicts in
the different fonts . We do this with the FDSelect array, and by
assuming that the same glyph will reference  matching FontDicts in
each source font. We return a mapping from fdIndex in the default
font to a dictionary which maps each master list index of each
region font to the equivalent fdIndex in the region font.

### Function: merge_region_fonts(varFont, model, ordered_fonts_list, glyphOrder)

### Function: _get_cs(charstrings, glyphName, filterEmpty)

### Function: _add_new_vsindex(model, key, masterSupports, vsindex_dict, vsindex_by_key, varDataList)

### Function: merge_charstrings(glyphOrder, num_masters, top_dicts, masterModel)

## Class: CFFToCFF2OutlineExtractor

**Description:** This class is used to remove the initial width from the CFF
charstring without trying to add the width to self.nominalWidthX,
which is None.

## Class: MergeOutlineExtractor

**Description:** Used to extract the charstring commands - including hints - from a
CFF charstring in order to merge it as another set of region data
into a CFF2 variable font charstring.

## Class: CFF2CharStringMergePen

**Description:** Pen to merge Type 2 CharStrings.

### Function: popallWidth(self, evenOdd)

### Function: __init__(self, pen, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private, blender)

### Function: countHints(self)

### Function: _hint_op(self, type, args)

### Function: op_hstem(self, index)

### Function: op_vstem(self, index)

### Function: op_hstemhm(self, index)

### Function: op_vstemhm(self, index)

### Function: _get_hintmask(self, index)

### Function: op_hintmask(self, index)

### Function: op_cntrmask(self, index)

### Function: __init__(self, default_commands, glyphName, num_masters, master_idx, roundTolerance)

### Function: add_point(self, point_type, pt_coords)

### Function: add_hint(self, hint_type, args)

### Function: add_hintmask(self, hint_type, abs_args)

### Function: _moveTo(self, pt)

### Function: _lineTo(self, pt)

### Function: _curveToOne(self, pt1, pt2, pt3)

### Function: _closePath(self)

### Function: _endPath(self)

### Function: restart(self, region_idx)

### Function: getCommands(self)

### Function: reorder_blend_args(self, commands, get_delta_func)

**Description:** We first re-order the master coordinate values.
For a moveto to lineto, the args are now arranged as::

        [ [master_0 x,y], [master_1 x,y], [master_2 x,y] ]

We re-arrange this to::

        [       [master_0 x, master_1 x, master_2 x],
                [master_0 y, master_1 y, master_2 y]
        ]

If the master values are all the same, we collapse the list to
as single value instead of a list.

We then convert this to::

        [ [master_0 x] + [x delta tuple] + [numBlends=1]
          [master_0 y] + [y delta tuple] + [numBlends=1]
        ]

### Function: getCharString(self, private, globalSubrs, var_model, optimize)
