## AI Summary

A file named _afm.py.


### Function: _to_int(x)

### Function: _to_float(x)

### Function: _to_str(x)

### Function: _to_list_of_ints(s)

### Function: _to_list_of_floats(s)

### Function: _to_bool(s)

### Function: _parse_header(fh)

**Description:** Read the font metrics header (up to the char metrics) and returns
a dictionary mapping *key* to *val*.  *val* will be converted to the
appropriate python type as necessary; e.g.:

    * 'False'->False
    * '0'->0
    * '-168 -218 1000 898'-> [-168, -218, 1000, 898]

Dictionary keys are

  StartFontMetrics, FontName, FullName, FamilyName, Weight,
  ItalicAngle, IsFixedPitch, FontBBox, UnderlinePosition,
  UnderlineThickness, Version, Notice, EncodingScheme, CapHeight,
  XHeight, Ascender, Descender, StartCharMetrics

### Function: _parse_char_metrics(fh)

**Description:** Parse the given filehandle for character metrics information and return
the information as dicts.

It is assumed that the file cursor is on the line behind
'StartCharMetrics'.

Returns
-------
ascii_d : dict
     A mapping "ASCII num of the character" to `.CharMetrics`.
name_d : dict
     A mapping "character name" to `.CharMetrics`.

Notes
-----
This function is incomplete per the standard, but thus far parses
all the sample afm files tried.

### Function: _parse_kern_pairs(fh)

**Description:** Return a kern pairs dictionary; keys are (*char1*, *char2*) tuples and
values are the kern pair value.  For example, a kern pairs line like
``KPX A y -50``

will be represented as::

  d[ ('A', 'y') ] = -50

### Function: _parse_composites(fh)

**Description:** Parse the given filehandle for composites information return them as a
dict.

It is assumed that the file cursor is on the line behind 'StartComposites'.

Returns
-------
dict
    A dict mapping composite character names to a parts list. The parts
    list is a list of `.CompositePart` entries describing the parts of
    the composite.

Examples
--------
A composite definition line::

  CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;

will be represented as::

  composites['Aacute'] = [CompositePart(name='A', dx=0, dy=0),
                          CompositePart(name='acute', dx=160, dy=170)]

### Function: _parse_optional(fh)

**Description:** Parse the optional fields for kern pair data and composites.

Returns
-------
kern_data : dict
    A dict containing kerning information. May be empty.
    See `._parse_kern_pairs`.
composites : dict
    A dict containing composite information. May be empty.
    See `._parse_composites`.

## Class: AFM

### Function: __init__(self, fh)

**Description:** Parse the AFM file in file object *fh*.

### Function: get_bbox_char(self, c, isord)

### Function: string_width_height(self, s)

**Description:** Return the string width (including kerning) and string height
as a (*w*, *h*) tuple.

### Function: get_str_bbox_and_descent(self, s)

**Description:** Return the string bounding box and the maximal descent.

### Function: get_str_bbox(self, s)

**Description:** Return the string bounding box.

### Function: get_name_char(self, c, isord)

**Description:** Get the name of the character, i.e., ';' is 'semicolon'.

### Function: get_width_char(self, c, isord)

**Description:** Get the width of the character from the character metric WX field.

### Function: get_width_from_char_name(self, name)

**Description:** Get the width of the character from a type1 character name.

### Function: get_height_char(self, c, isord)

**Description:** Get the bounding box (ink) height of character *c* (space is 0).

### Function: get_kern_dist(self, c1, c2)

**Description:** Return the kerning pair distance (possibly 0) for chars *c1* and *c2*.

### Function: get_kern_dist_from_name(self, name1, name2)

**Description:** Return the kerning pair distance (possibly 0) for chars
*name1* and *name2*.

### Function: get_fontname(self)

**Description:** Return the font name, e.g., 'Times-Roman'.

### Function: postscript_name(self)

### Function: get_fullname(self)

**Description:** Return the font full name, e.g., 'Times-Roman'.

### Function: get_familyname(self)

**Description:** Return the font family name, e.g., 'Times'.

### Function: family_name(self)

**Description:** The font family name, e.g., 'Times'.

### Function: get_weight(self)

**Description:** Return the font weight, e.g., 'Bold' or 'Roman'.

### Function: get_angle(self)

**Description:** Return the fontangle as float.

### Function: get_capheight(self)

**Description:** Return the cap height as float.

### Function: get_xheight(self)

**Description:** Return the xheight as float.

### Function: get_underline_thickness(self)

**Description:** Return the underline thickness as float.

### Function: get_horizontal_stem_width(self)

**Description:** Return the standard horizontal stem width as float, or *None* if
not specified in AFM file.

### Function: get_vertical_stem_width(self)

**Description:** Return the standard vertical stem width as float, or *None* if
not specified in AFM file.
