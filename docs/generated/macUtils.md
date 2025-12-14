## AI Summary

A file named macUtils.py.


### Function: getSFNTResIndices(path)

**Description:** Determine whether a file has a 'sfnt' resource fork or not.

### Function: openTTFonts(path)

**Description:** Given a pathname, return a list of TTFont objects. In the case
of a flat TTF/OTF file, the list will contain just one font object;
but in the case of a Mac font suitcase it will contain as many
font objects as there are sfnt resources in the file.

## Class: SFNTResourceReader

**Description:** Simple read-only file wrapper for 'sfnt' resources.

### Function: __init__(self, path, res_name_or_index)
