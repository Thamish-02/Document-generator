## AI Summary

A file named glifLib.py.


## Class: GLIFFormatVersion

**Description:** Class representing the versions of the .glif format supported by the UFO version in use.

For a given :mod:`fontTools.ufoLib.UFOFormatVersion`, the :func:`supported_versions` method will
return the supported versions of the GLIF file format. If the UFO version is unspecified, the
:func:`supported_versions` method will return all available GLIF format versions.

## Class: Glyph

**Description:** Minimal glyph object. It has no glyph attributes until either
the draw() or the drawPoints() method has been called.

## Class: GlyphSet

**Description:** GlyphSet manages a set of .glif files inside one directory.

GlyphSet's constructor takes a path to an existing directory as it's
first argument. Reading glyph data can either be done through the
readGlyph() method, or by using GlyphSet's dictionary interface, where
the keys are glyph names and the values are (very) simple glyph objects.

To write a glyph to the glyph set, you use the writeGlyph() method.
The simple glyph objects returned through the dict interface do not
support writing, they are just a convenient way to get at the glyph data.

### Function: glyphNameToFileName(glyphName, existingFileNames)

**Description:** Wrapper around the userNameToFileName function in filenames.py

Note that existingFileNames should be a set for large glyphsets
or performance will suffer.

### Function: readGlyphFromString(aString, glyphObject, pointPen, formatVersions, validate)

**Description:** Read .glif data from a string into a glyph object.

The 'glyphObject' argument can be any kind of object (even None);
the readGlyphFromString() method will attempt to set the following
attributes on it:

width
        the advance width of the glyph
height
        the advance height of the glyph
unicodes
        a list of unicode values for this glyph
note
        a string
lib
        a dictionary containing custom data
image
        a dictionary containing image data
guidelines
        a list of guideline data dictionaries
anchors
        a list of anchor data dictionaries

All attributes are optional, in two ways:

1) An attribute *won't* be set if the .glif file doesn't
   contain data for it. 'glyphObject' will have to deal
   with default values itself.
2) If setting the attribute fails with an AttributeError
   (for example if the 'glyphObject' attribute is read-
   only), readGlyphFromString() will not propagate that
   exception, but ignore that attribute.

To retrieve outline information, you need to pass an object
conforming to the PointPen protocol as the 'pointPen' argument.
This argument may be None if you don't need the outline data.

The formatVersions optional argument define the GLIF format versions
that are allowed to be read.
The type is Optional[Iterable[tuple[int, int], int]]. It can contain
either integers (for the major versions to be allowed, with minor
digits defaulting to 0), or tuples of integers to specify both
(major, minor) versions.
By default when formatVersions is None all the GLIF format versions
currently defined are allowed to be read.

``validate`` will validate the read data. It is set to ``True`` by default.

### Function: _writeGlyphToBytes(glyphName, glyphObject, drawPointsFunc, writer, formatVersion, validate)

**Description:** Return .glif data for a glyph as a UTF-8 encoded bytes string.

### Function: writeGlyphToString(glyphName, glyphObject, drawPointsFunc, formatVersion, validate)

**Description:** Return .glif data for a glyph as a string. The XML declaration's
encoding is always set to "UTF-8".
The 'glyphObject' argument can be any kind of object (even None);
the writeGlyphToString() method will attempt to get the following
attributes from it:

width
        the advance width of the glyph
height
        the advance height of the glyph
unicodes
        a list of unicode values for this glyph
note
        a string
lib
        a dictionary containing custom data
image
        a dictionary containing image data
guidelines
        a list of guideline data dictionaries
anchors
        a list of anchor data dictionaries

All attributes are optional: if 'glyphObject' doesn't
have the attribute, it will simply be skipped.

To write outline data to the .glif file, writeGlyphToString() needs
a function (any callable object actually) that will take one
argument: an object that conforms to the PointPen protocol.
The function will be called by writeGlyphToString(); it has to call the
proper PointPen methods to transfer the outline to the .glif file.

The GLIF format version can be specified with the formatVersion argument.
This accepts either a tuple of integers for (major, minor), or a single
integer for the major digit only (with minor digit implied as 0).
By default when formatVesion is None the latest GLIF format version will
be used; currently it's 2.0, which is equivalent to formatVersion=(2, 0).

An UnsupportedGLIFFormat exception is raised if the requested UFO
formatVersion is not supported.

``validate`` will validate the written data. It is set to ``True`` by default.

### Function: _writeAdvance(glyphObject, element, validate)

### Function: _writeUnicodes(glyphObject, element, validate)

### Function: _writeNote(glyphObject, element, validate)

### Function: _writeImage(glyphObject, element, validate)

### Function: _writeGuidelines(glyphObject, element, identifiers, validate)

### Function: _writeAnchorsFormat1(pen, anchors, validate)

### Function: _writeAnchors(glyphObject, element, identifiers, validate)

### Function: _writeLib(glyphObject, element, validate)

### Function: validateLayerInfoVersion3ValueForAttribute(attr, value)

**Description:** This performs very basic validation of the value for attribute
following the UFO 3 fontinfo.plist specification. The results
of this should not be interpretted as *correct* for the font
that they are part of. This merely indicates that the value
is of the proper type and, where the specification defines
a set range of possible values for an attribute, that the
value is in the accepted range.

### Function: validateLayerInfoVersion3Data(infoData)

**Description:** This performs very basic validation of the value for infoData
following the UFO 3 layerinfo.plist specification. The results
of this should not be interpretted as *correct* for the font
that they are part of. This merely indicates that the values
are of the proper type and, where the specification defines
a set range of possible values for an attribute, that the
value is in the accepted range.

### Function: _glifTreeFromFile(aFile)

### Function: _glifTreeFromString(aString)

### Function: _readGlyphFromTree(tree, glyphObject, pointPen, formatVersions, validate)

### Function: _readGlyphFromTreeFormat1(tree, glyphObject, pointPen, validate)

### Function: _readGlyphFromTreeFormat2(tree, glyphObject, pointPen, validate, formatMinor)

### Function: _readName(glyphObject, root, validate)

### Function: _readAdvance(glyphObject, advance)

### Function: _readNote(glyphObject, note)

### Function: _readLib(glyphObject, lib, validate)

### Function: _readImage(glyphObject, image, validate)

### Function: buildOutlineFormat1(glyphObject, pen, outline, validate)

### Function: _buildAnchorFormat1(point, validate)

### Function: _buildOutlineContourFormat1(pen, contour, validate)

### Function: _buildOutlinePointsFormat1(pen, contour)

### Function: _buildOutlineComponentFormat1(pen, component, validate)

### Function: buildOutlineFormat2(glyphObject, pen, outline, identifiers, validate)

### Function: _buildOutlineContourFormat2(pen, contour, identifiers, validate)

### Function: _buildOutlinePointsFormat2(pen, contour, identifiers, validate)

### Function: _buildOutlineComponentFormat2(pen, component, identifiers, validate)

### Function: _validateAndMassagePointStructures(contour, pointAttributes, openContourOffCurveLeniency, validate)

### Function: _relaxedSetattr(object, attr, value)

### Function: _number(s)

**Description:** Given a numeric string, return an integer or a float, whichever
the string indicates. _number("1") will return the integer 1,
_number("1.0") will return the float 1.0.

>>> _number("1")
1
>>> _number("1.0")
1.0
>>> _number("a")  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
GlifLibError: Could not convert a to an int or float.

## Class: _DoneParsing

## Class: _BaseParser

### Function: _fetchUnicodes(glif)

**Description:** Get a list of unicodes listed in glif.

## Class: _FetchUnicodesParser

### Function: _fetchImageFileName(glif)

**Description:** The image file name (if any) from glif.

## Class: _FetchImageFileNameParser

### Function: _fetchComponentBases(glif)

**Description:** Get a list of component base glyphs listed in glif.

## Class: _FetchComponentBasesParser

## Class: GLIFPointPen

**Description:** Helper class using the PointPen protocol to write the <outline>
part of .glif files.

### Function: default(cls, ufoFormatVersion)

### Function: supported_versions(cls, ufoFormatVersion)

### Function: __init__(self, glyphName, glyphSet)

### Function: draw(self, pen, outputImpliedClosingLine)

**Description:** Draw this glyph onto a *FontTools* Pen.

### Function: drawPoints(self, pointPen)

**Description:** Draw this glyph onto a PointPen.

### Function: __init__(self, path, glyphNameToFileNameFunc, ufoFormatVersion, validateRead, validateWrite, expectContentsFile)

**Description:** 'path' should be a path (string) to an existing local directory, or
an instance of fs.base.FS class.

The optional 'glyphNameToFileNameFunc' argument must be a callback
function that takes two arguments: a glyph name and a list of all
existing filenames (if any exist). It should return a file name
(including the .glif extension). The glyphNameToFileName function
is called whenever a file name is created for a given glyph name.

``validateRead`` will validate read operations. Its default is ``True``.
``validateWrite`` will validate write operations. Its default is ``True``.
``expectContentsFile`` will raise a GlifLibError if a contents.plist file is
not found on the glyph set file system. This should be set to ``True`` if you
are reading an existing UFO and ``False`` if you create a fresh glyph set.

### Function: rebuildContents(self, validateRead)

**Description:** Rebuild the contents dict by loading contents.plist.

``validateRead`` will validate the data, by default it is set to the
class's ``validateRead`` value, can be overridden.

### Function: getReverseContents(self)

**Description:** Return a reversed dict of self.contents, mapping file names to
glyph names. This is primarily an aid for custom glyph name to file
name schemes that want to make sure they don't generate duplicate
file names. The file names are converted to lowercase so we can
reliably check for duplicates that only differ in case, which is
important for case-insensitive file systems.

### Function: writeContents(self)

**Description:** Write the contents.plist file out to disk. Call this method when
you're done writing glyphs.

### Function: readLayerInfo(self, info, validateRead)

**Description:** ``validateRead`` will validate the data, by default it is set to the
class's ``validateRead`` value, can be overridden.

### Function: writeLayerInfo(self, info, validateWrite)

**Description:** ``validateWrite`` will validate the data, by default it is set to the
class's ``validateWrite`` value, can be overridden.

### Function: getGLIF(self, glyphName)

**Description:** Get the raw GLIF text for a given glyph name. This only works
for GLIF files that are already on disk.

This method is useful in situations when the raw XML needs to be
read from a glyph set for a particular glyph before fully parsing
it into an object structure via the readGlyph method.

Raises KeyError if 'glyphName' is not in contents.plist, or
GlifLibError if the file associated with can't be found.

### Function: getGLIFModificationTime(self, glyphName)

**Description:** Returns the modification time for the GLIF file with 'glyphName', as
a floating point number giving the number of seconds since the epoch.
Return None if the associated file does not exist or the underlying
filesystem does not support getting modified times.
Raises KeyError if the glyphName is not in contents.plist.

### Function: readGlyph(self, glyphName, glyphObject, pointPen, validate)

**Description:** Read a .glif file for 'glyphName' from the glyph set. The
'glyphObject' argument can be any kind of object (even None);
the readGlyph() method will attempt to set the following
attributes on it:

width
        the advance width of the glyph
height
        the advance height of the glyph
unicodes
        a list of unicode values for this glyph
note
        a string
lib
        a dictionary containing custom data
image
        a dictionary containing image data
guidelines
        a list of guideline data dictionaries
anchors
        a list of anchor data dictionaries

All attributes are optional, in two ways:

1) An attribute *won't* be set if the .glif file doesn't
   contain data for it. 'glyphObject' will have to deal
   with default values itself.
2) If setting the attribute fails with an AttributeError
   (for example if the 'glyphObject' attribute is read-
   only), readGlyph() will not propagate that exception,
   but ignore that attribute.

To retrieve outline information, you need to pass an object
conforming to the PointPen protocol as the 'pointPen' argument.
This argument may be None if you don't need the outline data.

readGlyph() will raise KeyError if the glyph is not present in
the glyph set.

``validate`` will validate the data, by default it is set to the
class's ``validateRead`` value, can be overridden.

### Function: writeGlyph(self, glyphName, glyphObject, drawPointsFunc, formatVersion, validate)

**Description:** Write a .glif file for 'glyphName' to the glyph set. The
'glyphObject' argument can be any kind of object (even None);
the writeGlyph() method will attempt to get the following
attributes from it:

width
        the advance width of the glyph
height
        the advance height of the glyph
unicodes
        a list of unicode values for this glyph
note
        a string
lib
        a dictionary containing custom data
image
        a dictionary containing image data
guidelines
        a list of guideline data dictionaries
anchors
        a list of anchor data dictionaries

All attributes are optional: if 'glyphObject' doesn't
have the attribute, it will simply be skipped.

To write outline data to the .glif file, writeGlyph() needs
a function (any callable object actually) that will take one
argument: an object that conforms to the PointPen protocol.
The function will be called by writeGlyph(); it has to call the
proper PointPen methods to transfer the outline to the .glif file.

The GLIF format version will be chosen based on the ufoFormatVersion
passed during the creation of this object. If a particular format
version is desired, it can be passed with the formatVersion argument.
The formatVersion argument accepts either a tuple of integers for
(major, minor), or a single integer for the major digit only (with
minor digit implied as 0).

An UnsupportedGLIFFormat exception is raised if the requested GLIF
formatVersion is not supported.

``validate`` will validate the data, by default it is set to the
class's ``validateWrite`` value, can be overridden.

### Function: deleteGlyph(self, glyphName)

**Description:** Permanently delete the glyph from the glyph set on disk. Will
raise KeyError if the glyph is not present in the glyph set.

### Function: keys(self)

### Function: has_key(self, glyphName)

### Function: __len__(self)

### Function: __getitem__(self, glyphName)

### Function: getUnicodes(self, glyphNames)

**Description:** Return a dictionary that maps glyph names to lists containing
the unicode value[s] for that glyph, if any. This parses the .glif
files partially, so it is a lot faster than parsing all files completely.
By default this checks all glyphs, but a subset can be passed with glyphNames.

### Function: getComponentReferences(self, glyphNames)

**Description:** Return a dictionary that maps glyph names to lists containing the
base glyph name of components in the glyph. This parses the .glif
files partially, so it is a lot faster than parsing all files completely.
By default this checks all glyphs, but a subset can be passed with glyphNames.

### Function: getImageReferences(self, glyphNames)

**Description:** Return a dictionary that maps glyph names to the file name of the image
referenced by the glyph. This parses the .glif files partially, so it is a
lot faster than parsing all files completely.
By default this checks all glyphs, but a subset can be passed with glyphNames.

### Function: close(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, exc_tb)

### Function: __init__(self)

### Function: parse(self, text)

### Function: startElementHandler(self, name, attrs)

### Function: endElementHandler(self, name)

### Function: __init__(self)

### Function: startElementHandler(self, name, attrs)

### Function: __init__(self)

### Function: startElementHandler(self, name, attrs)

### Function: __init__(self)

### Function: startElementHandler(self, name, attrs)

### Function: endElementHandler(self, name)

### Function: __init__(self, element, formatVersion, identifiers, validate)

### Function: beginPath(self, identifier)

### Function: endPath(self)

### Function: addPoint(self, pt, segmentType, smooth, name, identifier)

### Function: addComponent(self, glyphName, transformation, identifier)
