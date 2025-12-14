## AI Summary

A file named fontBuilder.py.


## Class: FontBuilder

### Function: buildCmapSubTable(cmapping, format, platformID, platEncID)

### Function: addFvar(font, axes, instances)

### Function: __init__(self, unitsPerEm, font, isTTF, glyphDataFormat)

**Description:** Initialize a FontBuilder instance.

If the `font` argument is not given, a new `TTFont` will be
constructed, and `unitsPerEm` must be given. If `isTTF` is True,
the font will be a glyf-based TTF; if `isTTF` is False it will be
a CFF-based OTF.

The `glyphDataFormat` argument corresponds to the `head` table field
that defines the format of the TrueType `glyf` table (default=0).
TrueType glyphs historically can only contain quadratic splines and static
components, but there's a proposal to add support for cubic Bezier curves as well
as variable composites/components at
https://github.com/harfbuzz/boring-expansion-spec/blob/main/glyf1.md
You can experiment with the new features by setting `glyphDataFormat` to 1.
A ValueError is raised if `glyphDataFormat` is left at 0 but glyphs are added
that contain cubic splines or varcomposites. This is to prevent accidentally
creating fonts that are incompatible with existing TrueType implementations.

If `font` is given, it must be a `TTFont` instance and `unitsPerEm`
must _not_ be given. The `isTTF` and `glyphDataFormat` arguments will be ignored.

### Function: save(self, file)

**Description:** Save the font. The 'file' argument can be either a pathname or a
writable file object.

### Function: _initTableWithValues(self, tableTag, defaults, values)

### Function: _updateTableWithValues(self, tableTag, values)

### Function: setupHead(self)

**Description:** Create a new `head` table and initialize it with default values,
which can be overridden by keyword arguments.

### Function: updateHead(self)

**Description:** Update the head table with the fields and values passed as
keyword arguments.

### Function: setupGlyphOrder(self, glyphOrder)

**Description:** Set the glyph order for the font.

### Function: setupCharacterMap(self, cmapping, uvs, allowFallback)

**Description:** Build the `cmap` table for the font. The `cmapping` argument should
be a dict mapping unicode code points as integers to glyph names.

The `uvs` argument, when passed, must be a list of tuples, describing
Unicode Variation Sequences. These tuples have three elements:
    (unicodeValue, variationSelector, glyphName)
`unicodeValue` and `variationSelector` are integer code points.
`glyphName` may be None, to indicate this is the default variation.
Text processors will then use the cmap to find the glyph name.
Each Unicode Variation Sequence should be an officially supported
sequence, but this is not policed.

### Function: setupNameTable(self, nameStrings, windows, mac)

**Description:** Create the `name` table for the font. The `nameStrings` argument must
be a dict, mapping nameIDs or descriptive names for the nameIDs to name
record values. A value is either a string, or a dict, mapping language codes
to strings, to allow localized name table entries.

By default, both Windows (platformID=3) and Macintosh (platformID=1) name
records are added, unless any of `windows` or `mac` arguments is False.

The following descriptive names are available for nameIDs:

    copyright (nameID 0)
    familyName (nameID 1)
    styleName (nameID 2)
    uniqueFontIdentifier (nameID 3)
    fullName (nameID 4)
    version (nameID 5)
    psName (nameID 6)
    trademark (nameID 7)
    manufacturer (nameID 8)
    designer (nameID 9)
    description (nameID 10)
    vendorURL (nameID 11)
    designerURL (nameID 12)
    licenseDescription (nameID 13)
    licenseInfoURL (nameID 14)
    typographicFamily (nameID 16)
    typographicSubfamily (nameID 17)
    compatibleFullName (nameID 18)
    sampleText (nameID 19)
    postScriptCIDFindfontName (nameID 20)
    wwsFamilyName (nameID 21)
    wwsSubfamilyName (nameID 22)
    lightBackgroundPalette (nameID 23)
    darkBackgroundPalette (nameID 24)
    variationsPostScriptNamePrefix (nameID 25)

### Function: setupOS2(self)

**Description:** Create a new `OS/2` table and initialize it with default values,
which can be overridden by keyword arguments.

### Function: setupCFF(self, psName, fontInfo, charStringsDict, privateDict)

### Function: setupCFF2(self, charStringsDict, fdArrayList, regions)

### Function: setupCFF2Regions(self, regions)

### Function: setupGlyf(self, glyphs, calcGlyphBounds, validateGlyphFormat)

**Description:** Create the `glyf` table from a dict, that maps glyph names
to `fontTools.ttLib.tables._g_l_y_f.Glyph` objects, for example
as made by `fontTools.pens.ttGlyphPen.TTGlyphPen`.

If `calcGlyphBounds` is True, the bounds of all glyphs will be
calculated. Only pass False if your glyph objects already have
their bounding box values set.

If `validateGlyphFormat` is True, raise ValueError if any of the glyphs contains
cubic curves or is a variable composite but head.glyphDataFormat=0.
Set it to False to skip the check if you know in advance all the glyphs are
compatible with the specified glyphDataFormat.

### Function: setupFvar(self, axes, instances)

**Description:** Adds an font variations table to the font.

Args:
    axes (list): See below.
    instances (list): See below.

``axes`` should be a list of axes, with each axis either supplied as
a py:class:`.designspaceLib.AxisDescriptor` object, or a tuple in the
format ```tupletag, minValue, defaultValue, maxValue, name``.
The ``name`` is either a string, or a dict, mapping language codes
to strings, to allow localized name table entries.

```instances`` should be a list of instances, with each instance either
supplied as a py:class:`.designspaceLib.InstanceDescriptor` object, or a
dict with keys ``location`` (mapping of axis tags to float values),
``stylename`` and (optionally) ``postscriptfontname``.
The ``stylename`` is either a string, or a dict, mapping language codes
to strings, to allow localized name table entries.

### Function: setupAvar(self, axes, mappings)

**Description:** Adds an axis variations table to the font.

Args:
    axes (list): A list of py:class:`.designspaceLib.AxisDescriptor` objects.

### Function: setupGvar(self, variations)

### Function: setupGVAR(self, variations)

### Function: calcGlyphBounds(self)

**Description:** Calculate the bounding boxes of all glyphs in the `glyf` table.
This is usually not called explicitly by client code.

### Function: setupHorizontalMetrics(self, metrics)

**Description:** Create a new `hmtx` table, for horizontal metrics.

The `metrics` argument must be a dict, mapping glyph names to
`(width, leftSidebearing)` tuples.

### Function: setupVerticalMetrics(self, metrics)

**Description:** Create a new `vmtx` table, for horizontal metrics.

The `metrics` argument must be a dict, mapping glyph names to
`(height, topSidebearing)` tuples.

### Function: setupMetrics(self, tableTag, metrics)

**Description:** See `setupHorizontalMetrics()` and `setupVerticalMetrics()`.

### Function: setupHorizontalHeader(self)

**Description:** Create a new `hhea` table initialize it with default values,
which can be overridden by keyword arguments.

### Function: setupVerticalHeader(self)

**Description:** Create a new `vhea` table initialize it with default values,
which can be overridden by keyword arguments.

### Function: setupVerticalOrigins(self, verticalOrigins, defaultVerticalOrigin)

**Description:** Create a new `VORG` table. The `verticalOrigins` argument must be
a dict, mapping glyph names to vertical origin values.

The `defaultVerticalOrigin` argument should be the most common vertical
origin value. If omitted, this value will be derived from the actual
values in the `verticalOrigins` argument.

### Function: setupPost(self, keepGlyphNames)

**Description:** Create a new `post` table and initialize it with default values,
which can be overridden by keyword arguments.

### Function: setupMaxp(self)

**Description:** Create a new `maxp` table. This is called implicitly by FontBuilder
itself and is usually not called by client code.

### Function: setupDummyDSIG(self)

**Description:** This adds an empty DSIG table to the font to make some MS applications
happy. This does not properly sign the font.

### Function: addOpenTypeFeatures(self, features, filename, tables, debug)

**Description:** Add OpenType features to the font from a string containing
Feature File syntax.

The `filename` argument is used in error messages and to determine
where to look for "include" files.

The optional `tables` argument can be a list of OTL tables tags to
build, allowing the caller to only build selected OTL tables. See
`fontTools.feaLib` for details.

The optional `debug` argument controls whether to add source debugging
information to the font in the `Debg` table.

### Function: addFeatureVariations(self, conditionalSubstitutions, featureTag)

**Description:** Add conditional substitutions to a Variable Font.

See `fontTools.varLib.featureVars.addFeatureVariations`.

### Function: setupCOLR(self, colorLayers, version, varStore, varIndexMap, clipBoxes, allowLayerReuse)

**Description:** Build new COLR table using color layers dictionary.

Cf. `fontTools.colorLib.builder.buildCOLR`.

### Function: setupCPAL(self, palettes, paletteTypes, paletteLabels, paletteEntryLabels)

**Description:** Build new CPAL table using list of palettes.

Optionally build CPAL v1 table using paletteTypes, paletteLabels and
paletteEntryLabels.

Cf. `fontTools.colorLib.builder.buildCPAL`.

### Function: setupStat(self, axes, locations, elidedFallbackName)

**Description:** Build a new 'STAT' table.

See `fontTools.otlLib.builder.buildStatTable` for details about
the arguments.
