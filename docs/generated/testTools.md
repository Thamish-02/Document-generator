## AI Summary

A file named testTools.py.


### Function: parseXML(xmlSnippet)

**Description:** Parses a snippet of XML.

Input can be either a single string (unicode or UTF-8 bytes), or a
a sequence of strings.

The result is in the same format that would be returned by
XMLReader, but the parser imposes no constraints on the root
element so it can be called on small snippets of TTX files.

### Function: parseXmlInto(font, parseInto, xmlSnippet)

## Class: FakeFont

## Class: TestXMLReader_

### Function: makeXMLWriter(newlinestr)

### Function: getXML(func, ttFont)

**Description:** Call the passed toXML function and return the written content as a
list of lines (unicode strings).
Result is stripped of XML declaration and OS-specific newline characters.

### Function: stripVariableItemsFromTTX(string, ttLibVersion, checkSumAdjustment, modified, created, sfntVersion)

**Description:** Strip stuff like ttLibVersion, checksums, timestamps, etc. from TTX dumps.

## Class: MockFont

**Description:** A font-like object that automatically adds any looked up glyphname
to its glyphOrder.

## Class: TestCase

## Class: DataFilesHandler

### Function: __init__(self, glyphs)

### Function: __contains__(self, tag)

### Function: __getitem__(self, tag)

### Function: __setitem__(self, tag, table)

### Function: get(self, tag, default)

### Function: getGlyphID(self, name)

### Function: getGlyphIDMany(self, lst)

### Function: getGlyphName(self, glyphID)

### Function: getGlyphNameMany(self, lst)

### Function: getGlyphOrder(self)

### Function: getReverseGlyphMap(self)

### Function: getGlyphNames(self)

### Function: __init__(self)

### Function: startElement_(self, name, attrs)

### Function: endElement_(self, name)

### Function: addCharacterData_(self, data)

### Function: __init__(self)

### Function: getGlyphID(self, glyph)

### Function: getReverseGlyphMap(self)

### Function: getGlyphName(self, gid)

### Function: getGlyphOrder(self)

### Function: __init__(self, methodName)

### Function: setUp(self)

### Function: tearDown(self)

### Function: getpath(self, testfile)

### Function: temp_dir(self)

### Function: temp_font(self, font_path, file_name)

## Class: AllocatingDict

### Function: __missing__(reverseDict, key)
