## AI Summary

A file named afmLib.py.


## Class: error

## Class: AFM

### Function: readlines(path)

### Function: writelines(path, lines, sep)

### Function: __init__(self, path)

**Description:** AFM file reader.

Instantiating an object with a path name will cause the file to be opened,
read, and parsed. Alternatively the path can be left unspecified, and a
file can be parsed later with the :meth:`read` method.

### Function: read(self, path)

**Description:** Opens, reads and parses a file.

### Function: parsechar(self, rest)

### Function: parsekernpair(self, rest)

### Function: parseattr(self, word, rest)

### Function: parsecomposite(self, rest)

### Function: write(self, path, sep)

**Description:** Writes out an AFM font to the given path.

### Function: has_kernpair(self, pair)

**Description:** Returns `True` if the given glyph pair (specified as a tuple) exists
in the kerning dictionary.

### Function: kernpairs(self)

**Description:** Returns a list of all kern pairs in the kerning dictionary.

### Function: has_char(self, char)

**Description:** Returns `True` if the given glyph exists in the font.

### Function: chars(self)

**Description:** Returns a list of all glyph names in the font.

### Function: comments(self)

**Description:** Returns all comments from the file.

### Function: addComment(self, comment)

**Description:** Adds a new comment to the file.

### Function: addComposite(self, glyphName, components)

**Description:** Specifies that the glyph `glyphName` is made up of the given components.
The components list should be of the following form::

        [
                (glyphname, xOffset, yOffset),
                ...
        ]

### Function: __getattr__(self, attr)

### Function: __setattr__(self, attr, value)

### Function: __delattr__(self, attr)

### Function: __getitem__(self, key)

### Function: __setitem__(self, key, value)

### Function: __delitem__(self, key)

### Function: __repr__(self)

### Function: myKey(a)

**Description:** Custom key function to make sure unencoded chars (-1)
end up at the end of the list after sorting.
