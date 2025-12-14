## AI Summary

A file named dviread.py.


## Class: Text

**Description:** A glyph in the dvi file.

The *x* and *y* attributes directly position the glyph.  The *font*,
*glyph*, and *width* attributes are kept public for back-compatibility,
but users wanting to draw the glyph themselves are encouraged to instead
load the font specified by `font_path` at `font_size`, warp it with the
effects specified by `font_effects`, and load the glyph specified by
`glyph_name_or_index`.

### Function: _dispatch(table, min, max, state, args)

**Description:** Decorator for dispatch by opcode. Sets the values in *table*
from *min* to *max* to this method, adds a check that the Dvi state
matches *state* if not None, reads arguments from the file according
to *args*.

Parameters
----------
table : dict[int, callable]
    The dispatch table to be filled in.

min, max : int
    Range of opcodes that calls the registered function; *max* defaults to
    *min*.

state : _dvistate, optional
    State of the Dvi object in which these opcodes are allowed.

args : list[str], default: ['raw']
    Sequence of argument specifications:

    - 'raw': opcode minus minimum
    - 'u1': read one unsigned byte
    - 'u4': read four bytes, treat as an unsigned number
    - 's4': read four bytes, treat as a signed number
    - 'slen': read (opcode - minimum) bytes, treat as signed
    - 'slen1': read (opcode - minimum + 1) bytes, treat as signed
    - 'ulen1': read (opcode - minimum + 1) bytes, treat as unsigned
    - 'olen1': read (opcode - minimum + 1) bytes, treat as unsigned
      if under four bytes, signed if four bytes

## Class: Dvi

**Description:** A reader for a dvi ("device-independent") file, as produced by TeX.

The current implementation can only iterate through pages in order,
and does not even attempt to verify the postamble.

This class can be used as a context manager to close the underlying
file upon exit. Pages can be read via iteration. Here is an overly
simple way to extract text without trying to detect whitespace::

    >>> with matplotlib.dviread.Dvi('input.dvi', 72) as dvi:
    ...     for page in dvi:
    ...         print(''.join(chr(t.glyph) for t in page.text))

## Class: DviFont

**Description:** Encapsulation of a font that a DVI file can refer to.

This class holds a font's texname and size, supports comparison,
and knows the widths of glyphs in the same units as the AFM file.
There are also internal attributes (for use by dviread.py) that
are *not* used for comparison.

The size is in Adobe points (converted from TeX points).

Parameters
----------
scale : float
    Factor by which the font is scaled from its natural size.
tfm : Tfm
    TeX font metrics for this font
texname : bytes
   Name of the font as used internally by TeX and friends, as an ASCII
   bytestring.  This is usually very different from any external font
   names; `PsfontsMap` can be used to find the external name of the font.
vf : Vf
   A TeX "virtual font" file, or None if this font is not virtual.

Attributes
----------
texname : bytes
size : float
   Size of the font in Adobe points, converted from the slightly
   smaller TeX points.
widths : list
   Widths of glyphs in glyph-space units, typically 1/1000ths of
   the point size.

## Class: Vf

**Description:** A virtual font (\*.vf file) containing subroutines for dvi files.

Parameters
----------
filename : str or path-like

Notes
-----
The virtual font format is a derivative of dvi:
http://mirrors.ctan.org/info/knuth/virtual-fonts
This class reuses some of the machinery of `Dvi`
but replaces the `_read` loop and dispatch mechanism.

Examples
--------
::

    vf = Vf(filename)
    glyph = vf[code]
    glyph.text, glyph.boxes, glyph.width

### Function: _mul2012(num1, num2)

**Description:** Multiply two numbers in 20.12 fixed point format.

## Class: Tfm

**Description:** A TeX Font Metric file.

This implementation covers only the bare minimum needed by the Dvi class.

Parameters
----------
filename : str or path-like

Attributes
----------
checksum : int
   Used for verifying against the dvi file.
design_size : int
   Design size of the font (unknown units)
width, height, depth : dict
   Dimensions of each character, need to be scaled by the factor
   specified in the dvi file. These are dicts because indexing may
   not start from 0.

## Class: PsfontsMap

**Description:** A psfonts.map formatted file, mapping TeX fonts to PS fonts.

Parameters
----------
filename : str or path-like

Notes
-----
For historical reasons, TeX knows many Type-1 fonts by different
names than the outside world. (For one thing, the names have to
fit in eight characters.) Also, TeX's native fonts are not Type-1
but Metafont, which is nontrivial to convert to PostScript except
as a bitmap. While high-quality conversions to Type-1 format exist
and are shipped with modern TeX distributions, we need to know
which Type-1 fonts are the counterparts of which native fonts. For
these reasons a mapping is needed from internal font names to font
file names.

A texmf tree typically includes mapping files called e.g.
:file:`psfonts.map`, :file:`pdftex.map`, or :file:`dvipdfm.map`.
The file :file:`psfonts.map` is used by :program:`dvips`,
:file:`pdftex.map` by :program:`pdfTeX`, and :file:`dvipdfm.map`
by :program:`dvipdfm`. :file:`psfonts.map` might avoid embedding
the 35 PostScript fonts (i.e., have no filename for them, as in
the Times-Bold example above), while the pdf-related files perhaps
only avoid the "Base 14" pdf fonts. But the user may have
configured these files differently.

Examples
--------
>>> map = PsfontsMap(find_tex_file('pdftex.map'))
>>> entry = map[b'ptmbo8r']
>>> entry.texname
b'ptmbo8r'
>>> entry.psname
b'Times-Bold'
>>> entry.encoding
'/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc'
>>> entry.effects
{'slant': 0.16700000000000001}
>>> entry.filename

### Function: _parse_enc(path)

**Description:** Parse a \*.enc file referenced from a psfonts.map style file.

The format supported by this function is a tiny subset of PostScript.

Parameters
----------
path : `os.PathLike`

Returns
-------
list
    The nth entry of the list is the PostScript glyph name of the nth
    glyph.

## Class: _LuatexKpsewhich

### Function: find_tex_file(filename)

**Description:** Find a file in the texmf tree using kpathsea_.

The kpathsea library, provided by most existing TeX distributions, both
on Unix-like systems and on Windows (MikTeX), is invoked via a long-lived
luatex process if luatex is installed, or via kpsewhich otherwise.

.. _kpathsea: https://www.tug.org/kpathsea/

Parameters
----------
filename : str or path-like

Raises
------
FileNotFoundError
    If the file is not found.

### Function: _fontfile(cls, suffix, texname)

### Function: _get_pdftexmap_entry(self)

### Function: font_path(self)

**Description:** The `~pathlib.Path` to the font for this glyph.

### Function: font_size(self)

**Description:** The font size.

### Function: font_effects(self)

**Description:** The "font effects" dict for this glyph.

This dict contains the values for this glyph of SlantFont and
ExtendFont (if any), read off :file:`pdftex.map`.

### Function: glyph_name_or_index(self)

**Description:** Either the glyph name or the native charmap glyph index.

If :file:`pdftex.map` specifies an encoding for this glyph's font, that
is a mapping of glyph indices to Adobe glyph names; use it to convert
dvi indices to glyph names.  Callers can then convert glyph names to
glyph indices (with FT_Get_Name_Index/get_name_index), and load the
glyph using FT_Load_Glyph/load_glyph.

If :file:`pdftex.map` specifies no encoding, the indices directly map
to the font's "native" charmap; glyphs should directly load using
FT_Load_Char/load_char after selecting the native charmap.

### Function: decorate(method)

### Function: __init__(self, filename, dpi)

**Description:** Read the data from the file named *filename* and convert
TeX's internal units to units of *dpi* per inch.
*dpi* only sets the units and does not limit the resolution.
Use None to return TeX's internal units.

### Function: __enter__(self)

**Description:** Context manager enter method, does nothing.

### Function: __exit__(self, etype, evalue, etrace)

**Description:** Context manager exit method, closes the underlying file if it is open.

### Function: __iter__(self)

**Description:** Iterate through the pages of the file.

Yields
------
Page
    Details of all the text and box objects on the page.
    The Page tuple contains lists of Text and Box tuples and
    the page dimensions, and the Text and Box tuples contain
    coordinates transformed into a standard Cartesian
    coordinate system at the dpi value given when initializing.
    The coordinates are floating point numbers, but otherwise
    precision is not lost and coordinate values are not clipped to
    integers.

### Function: close(self)

**Description:** Close the underlying file if it is open.

### Function: _output(self)

**Description:** Output the text and boxes belonging to the most recent page.
page = dvi._output()

### Function: _read(self)

**Description:** Read one page from the file. Return True if successful,
False if there were no more pages.

### Function: _read_arg(self, nbytes, signed)

**Description:** Read and return a big-endian integer *nbytes* long.
Signedness is determined by the *signed* keyword.

### Function: _set_char_immediate(self, char)

### Function: _set_char(self, char)

### Function: _set_rule(self, a, b)

### Function: _put_char(self, char)

### Function: _put_char_real(self, char)

### Function: _put_rule(self, a, b)

### Function: _put_rule_real(self, a, b)

### Function: _nop(self, _)

### Function: _bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p)

### Function: _eop(self, _)

### Function: _push(self, _)

### Function: _pop(self, _)

### Function: _right(self, b)

### Function: _right_w(self, new_w)

### Function: _right_x(self, new_x)

### Function: _down(self, a)

### Function: _down_y(self, new_y)

### Function: _down_z(self, new_z)

### Function: _fnt_num_immediate(self, k)

### Function: _fnt_num(self, new_f)

### Function: _xxx(self, datalen)

### Function: _fnt_def(self, k, c, s, d, a, l)

### Function: _fnt_def_real(self, k, c, s, d, a, l)

### Function: _pre(self, i, num, den, mag, k)

### Function: _post(self, _)

### Function: _post_post(self, _)

### Function: _malformed(self, offset)

### Function: __init__(self, scale, tfm, texname, vf)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __repr__(self)

### Function: _width_of(self, char)

**Description:** Width of char in dvi units.

### Function: _height_depth_of(self, char)

**Description:** Height and depth of char in dvi units.

### Function: __init__(self, filename)

### Function: __getitem__(self, code)

### Function: _read(self)

**Description:** Read one page from the file. Return True if successful,
False if there were no more pages.

### Function: _init_packet(self, pl)

### Function: _finalize_packet(self, packet_char, packet_width)

### Function: _pre(self, i, x, cs, ds)

### Function: __init__(self, filename)

### Function: __new__(cls, filename)

### Function: __getitem__(self, texname)

### Function: _parse_and_cache_line(self, line)

**Description:** Parse a line in the font mapping file.

The format is (partially) documented at
http://mirrors.ctan.org/systems/doc/pdftex/manual/pdftex-a.pdf
https://tug.org/texinfohtml/dvips.html#psfonts_002emap
Each line can have the following fields:

- tfmname (first, only required field),
- psname (defaults to tfmname, must come immediately after tfmname if
  present),
- fontflags (integer, must come immediately after psname if present,
  ignored by us),
- special (SlantFont and ExtendFont, only field that is double-quoted),
- fontfile, encodingfile (optional, prefixed by <, <<, or <[; << always
  precedes a font, <[ always precedes an encoding, < can precede either
  but then an encoding file must have extension .enc; < and << also
  request different font subsetting behaviors but we ignore that; < can
  be separated from the filename by whitespace).

special, fontfile, and encodingfile can appear in any order.

### Function: __new__(cls)

### Function: _new_proc(self)

### Function: search(self, filename)

### Function: wrapper(self, byte)
