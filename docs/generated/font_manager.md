## AI Summary

A file named font_manager.py.


### Function: get_fontext_synonyms(fontext)

**Description:** Return a list of file extensions that are synonyms for
the given file extension *fileext*.

### Function: list_fonts(directory, extensions)

**Description:** Return a list of all fonts matching any of the extensions, found
recursively under the directory.

### Function: win32FontDirectory()

**Description:** Return the user-specified font directory for Win32.  This is
looked up from the registry key ::

  \\HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders\Fonts

If the key is not found, ``%WINDIR%\Fonts`` will be returned.

### Function: _get_win32_installed_fonts()

**Description:** List the font paths known to the Windows registry.

### Function: _get_fontconfig_fonts()

**Description:** Cache and list the font paths known to ``fc-list``.

### Function: _get_macos_fonts()

**Description:** Cache and list the font paths known to ``system_profiler SPFontsDataType``.

### Function: findSystemFonts(fontpaths, fontext)

**Description:** Search for fonts in the specified font paths.  If no paths are
given, will use a standard set of system paths, as well as the
list of fonts tracked by fontconfig if fontconfig is installed and
available.  A list of TrueType fonts are returned by default with
AFM fonts as an option.

## Class: FontEntry

**Description:** A class for storing Font properties.

It is used when populating the font lookup dictionary.

### Function: ttfFontProperty(font)

**Description:** Extract information from a TrueType font file.

Parameters
----------
font : `.FT2Font`
    The TrueType font file from which information will be extracted.

Returns
-------
`FontEntry`
    The extracted font properties.

### Function: afmFontProperty(fontpath, font)

**Description:** Extract information from an AFM font file.

Parameters
----------
fontpath : str
    The filename corresponding to *font*.
font : AFM
    The AFM font file from which information will be extracted.

Returns
-------
`FontEntry`
    The extracted font properties.

### Function: _cleanup_fontproperties_init(init_method)

**Description:** A decorator to limit the call signature to single a positional argument
or alternatively only keyword arguments.

We still accept but deprecate all other call signatures.

When the deprecation expires we can switch the signature to::

    __init__(self, pattern=None, /, *, family=None, style=None, ...)

plus a runtime check that pattern is not used alongside with the
keyword arguments. This results eventually in the two possible
call signatures::

    FontProperties(pattern)
    FontProperties(family=..., size=..., ...)

## Class: FontProperties

**Description:** A class for storing and manipulating font properties.

The font properties are the six properties described in the
`W3C Cascading Style Sheet, Level 1
<http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
specification and *math_fontfamily* for math fonts:

- family: A list of font names in decreasing order of priority.
  The items may include a generic font family name, either 'sans-serif',
  'serif', 'cursive', 'fantasy', or 'monospace'.  In that case, the actual
  font to be used will be looked up from the associated rcParam during the
  search process in `.findfont`. Default: :rc:`font.family`

- style: Either 'normal', 'italic' or 'oblique'.
  Default: :rc:`font.style`

- variant: Either 'normal' or 'small-caps'.
  Default: :rc:`font.variant`

- stretch: A numeric value in the range 0-1000 or one of
  'ultra-condensed', 'extra-condensed', 'condensed',
  'semi-condensed', 'normal', 'semi-expanded', 'expanded',
  'extra-expanded' or 'ultra-expanded'. Default: :rc:`font.stretch`

- weight: A numeric value in the range 0-1000 or one of
  'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
  'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
  'extra bold', 'black'. Default: :rc:`font.weight`

- size: Either a relative value of 'xx-small', 'x-small',
  'small', 'medium', 'large', 'x-large', 'xx-large' or an
  absolute font size, e.g., 10. Default: :rc:`font.size`

- math_fontfamily: The family of fonts used to render math text.
  Supported values are: 'dejavusans', 'dejavuserif', 'cm',
  'stix', 'stixsans' and 'custom'. Default: :rc:`mathtext.fontset`

Alternatively, a font may be specified using the absolute path to a font
file, by using the *fname* kwarg.  However, in this case, it is typically
simpler to just pass the path (as a `pathlib.Path`, not a `str`) to the
*font* kwarg of the `.Text` object.

The preferred usage of font sizes is to use the relative values,
e.g.,  'large', instead of absolute font sizes, e.g., 12.  This
approach allows all text sizes to be made larger or smaller based
on the font manager's default font size.

This class accepts a single positional string as fontconfig_ pattern_,
or alternatively individual properties as keyword arguments::

    FontProperties(pattern)
    FontProperties(*, family=None, style=None, variant=None, ...)

This support does not depend on fontconfig; we are merely borrowing its
pattern syntax for use here.

.. _fontconfig: https://www.freedesktop.org/wiki/Software/fontconfig/
.. _pattern:
   https://www.freedesktop.org/software/fontconfig/fontconfig-user.html

Note that Matplotlib's internal font manager and fontconfig use a
different algorithm to lookup fonts, so the results of the same pattern
may be different in Matplotlib than in other applications that use
fontconfig.

## Class: _JSONEncoder

### Function: _json_decode(o)

### Function: json_dump(data, filename)

**Description:** Dump `FontManager` *data* as JSON to the file named *filename*.

See Also
--------
json_load

Notes
-----
File paths that are children of the Matplotlib data path (typically, fonts
shipped with Matplotlib) are stored relative to that data path (to remain
valid across virtualenvs).

This function temporarily locks the output file to prevent multiple
processes from overwriting one another's output.

### Function: json_load(filename)

**Description:** Load a `FontManager` from the JSON file named *filename*.

See Also
--------
json_dump

## Class: FontManager

**Description:** On import, the `FontManager` singleton instance creates a list of ttf and
afm fonts and caches their `FontProperties`.  The `FontManager.findfont`
method does a nearest neighbor search to find the font that most closely
matches the specification.  If no good enough match is found, the default
font is returned.

Fonts added with the `FontManager.addfont` method will not persist in the
cache; therefore, `addfont` will need to be called every time Matplotlib is
imported. This method should only be used if and when a font cannot be
installed on your operating system by other means.

Notes
-----
The `FontManager.addfont` method must be called on the global `FontManager`
instance.

Example usage::

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_dirs = ["/resources/fonts"]  # The path to the custom font file.
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

### Function: is_opentype_cff_font(filename)

**Description:** Return whether the given font is a Postscript Compact Font Format Font
embedded in an OpenType wrapper.  Used by the PostScript and PDF backends
that cannot subset these fonts.

### Function: _get_font(font_filepaths, hinting_factor)

### Function: _cached_realpath(path)

### Function: get_font(font_filepaths, hinting_factor)

**Description:** Get an `.ft2font.FT2Font` object given a list of file paths.

Parameters
----------
font_filepaths : Iterable[str, Path, bytes], str, Path, bytes
    Relative or absolute paths to the font files to be used.

    If a single string, bytes, or `pathlib.Path`, then it will be treated
    as a list with that entry only.

    If more than one filepath is passed, then the returned FT2Font object
    will fall back through the fonts, in the order given, to find a needed
    glyph.

Returns
-------
`.ft2font.FT2Font`

### Function: _load_fontmanager()

### Function: _repr_html_(self)

### Function: _repr_png_(self)

### Function: get_weight()

### Function: wrapper(self)

### Function: __init__(self, family, style, variant, weight, stretch, size, fname, math_fontfamily)

### Function: _from_any(cls, arg)

**Description:** Generic constructor which can build a `.FontProperties` from any of the
following:

- a `.FontProperties`: it is passed through as is;
- `None`: a `.FontProperties` using rc values is used;
- an `os.PathLike`: it is used as path to the font file;
- a `str`: it is parsed as a fontconfig pattern;
- a `dict`: it is passed as ``**kwargs`` to `.FontProperties`.

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: __str__(self)

### Function: get_family(self)

**Description:** Return a list of individual font family names or generic family names.

The font families or generic font families (which will be resolved
from their respective rcParams when searching for a matching font) in
the order of preference.

### Function: get_name(self)

**Description:** Return the name of the font that best matches the font properties.

### Function: get_style(self)

**Description:** Return the font style.  Values are: 'normal', 'italic' or 'oblique'.

### Function: get_variant(self)

**Description:** Return the font variant.  Values are: 'normal' or 'small-caps'.

### Function: get_weight(self)

**Description:** Set the font weight.  Options are: A numeric value in the
range 0-1000 or one of 'light', 'normal', 'regular', 'book',
'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold',
'heavy', 'extra bold', 'black'

### Function: get_stretch(self)

**Description:** Return the font stretch or width.  Options are: 'ultra-condensed',
'extra-condensed', 'condensed', 'semi-condensed', 'normal',
'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'.

### Function: get_size(self)

**Description:** Return the font size.

### Function: get_file(self)

**Description:** Return the filename of the associated font.

### Function: get_fontconfig_pattern(self)

**Description:** Get a fontconfig_ pattern_ suitable for looking up the font as
specified with fontconfig's ``fc-match`` utility.

This support does not depend on fontconfig; we are merely borrowing its
pattern syntax for use here.

### Function: set_family(self, family)

**Description:** Change the font family.  Can be either an alias (generic name
is CSS parlance), such as: 'serif', 'sans-serif', 'cursive',
'fantasy', or 'monospace', a real font name or a list of real
font names.  Real font names are not supported when
:rc:`text.usetex` is `True`. Default: :rc:`font.family`

### Function: set_style(self, style)

**Description:** Set the font style.

Parameters
----------
style : {'normal', 'italic', 'oblique'}, default: :rc:`font.style`

### Function: set_variant(self, variant)

**Description:** Set the font variant.

Parameters
----------
variant : {'normal', 'small-caps'}, default: :rc:`font.variant`

### Function: set_weight(self, weight)

**Description:** Set the font weight.

Parameters
----------
weight : int or {'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'}, default: :rc:`font.weight`
    If int, must be in the range  0-1000.

### Function: set_stretch(self, stretch)

**Description:** Set the font stretch or width.

Parameters
----------
stretch : int or {'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'}, default: :rc:`font.stretch`
    If int, must be in the range  0-1000.

### Function: set_size(self, size)

**Description:** Set the font size.

Parameters
----------
size : float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, default: :rc:`font.size`
    If a float, the font size in points. The string values denote
    sizes relative to the default font size.

### Function: set_file(self, file)

**Description:** Set the filename of the fontfile to use.  In this case, all
other properties will be ignored.

### Function: set_fontconfig_pattern(self, pattern)

**Description:** Set the properties by parsing a fontconfig_ *pattern*.

This support does not depend on fontconfig; we are merely borrowing its
pattern syntax for use here.

### Function: get_math_fontfamily(self)

**Description:** Return the name of the font family used for math text.

The default font is :rc:`mathtext.fontset`.

### Function: set_math_fontfamily(self, fontfamily)

**Description:** Set the font family for text in math mode.

If not set explicitly, :rc:`mathtext.fontset` will be used.

Parameters
----------
fontfamily : str
    The name of the font family.

    Available font families are defined in the
    :ref:`default matplotlibrc file <customizing-with-matplotlibrc-files>`.

See Also
--------
.text.Text.get_math_fontfamily

### Function: copy(self)

**Description:** Return a copy of self.

### Function: default(self, o)

### Function: __init__(self, size, weight)

### Function: addfont(self, path)

**Description:** Cache the properties of the font at *path* to make it available to the
`FontManager`.  The type of font is inferred from the path suffix.

Parameters
----------
path : str or path-like

Notes
-----
This method is useful for adding a custom font without installing it in
your operating system. See the `FontManager` singleton instance for
usage and caveats about this function.

### Function: defaultFont(self)

### Function: get_default_weight(self)

**Description:** Return the default font weight.

### Function: get_default_size()

**Description:** Return the default font size.

### Function: set_default_weight(self, weight)

**Description:** Set the default font weight.  The initial value is 'normal'.

### Function: _expand_aliases(family)

### Function: score_family(self, families, family2)

**Description:** Return a match score between the list of font families in
*families* and the font family name *family2*.

An exact match at the head of the list returns 0.0.

A match further down the list will return between 0 and 1.

No match will return 1.0.

### Function: score_style(self, style1, style2)

**Description:** Return a match score between *style1* and *style2*.

An exact match returns 0.0.

A match between 'italic' and 'oblique' returns 0.1.

No match returns 1.0.

### Function: score_variant(self, variant1, variant2)

**Description:** Return a match score between *variant1* and *variant2*.

An exact match returns 0.0, otherwise 1.0.

### Function: score_stretch(self, stretch1, stretch2)

**Description:** Return a match score between *stretch1* and *stretch2*.

The result is the absolute value of the difference between the
CSS numeric values of *stretch1* and *stretch2*, normalized
between 0.0 and 1.0.

### Function: score_weight(self, weight1, weight2)

**Description:** Return a match score between *weight1* and *weight2*.

The result is 0.0 if both weight1 and weight 2 are given as strings
and have the same value.

Otherwise, the result is the absolute value of the difference between
the CSS numeric values of *weight1* and *weight2*, normalized between
0.05 and 1.0.

### Function: score_size(self, size1, size2)

**Description:** Return a match score between *size1* and *size2*.

If *size2* (the size specified in the font file) is 'scalable', this
function always returns 0.0, since any font size can be generated.

Otherwise, the result is the absolute distance between *size1* and
*size2*, normalized so that the usual range of font sizes (6pt -
72pt) will lie between 0.0 and 1.0.

### Function: findfont(self, prop, fontext, directory, fallback_to_default, rebuild_if_missing)

**Description:** Find the path to the font file most closely matching the given font properties.

Parameters
----------
prop : str or `~matplotlib.font_manager.FontProperties`
    The font properties to search for. This can be either a
    `.FontProperties` object or a string defining a
    `fontconfig patterns`_.

fontext : {'ttf', 'afm'}, default: 'ttf'
    The extension of the font file:

    - 'ttf': TrueType and OpenType fonts (.ttf, .ttc, .otf)
    - 'afm': Adobe Font Metrics (.afm)

directory : str, optional
    If given, only search this directory and its subdirectories.

fallback_to_default : bool
    If True, will fall back to the default font family (usually
    "DejaVu Sans" or "Helvetica") if the first lookup hard-fails.

rebuild_if_missing : bool
    Whether to rebuild the font cache and search again if the first
    match appears to point to a nonexisting font (i.e., the font cache
    contains outdated entries).

Returns
-------
str
    The filename of the best matching font.

Notes
-----
This performs a nearest neighbor search.  Each font is given a
similarity score to the target font properties.  The first font with
the highest score is returned.  If no matches below a certain
threshold are found, the default font (usually DejaVu Sans) is
returned.

The result is cached, so subsequent lookups don't have to
perform the O(n) nearest neighbor search.

See the `W3C Cascading Style Sheet, Level 1
<http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ documentation
for a description of the font finding algorithm.

.. _fontconfig patterns:
   https://www.freedesktop.org/software/fontconfig/fontconfig-user.html

### Function: get_font_names(self)

**Description:** Return the list of available fonts.

### Function: _find_fonts_by_props(self, prop, fontext, directory, fallback_to_default, rebuild_if_missing)

**Description:** Find the paths to the font files most closely matching the given properties.

Parameters
----------
prop : str or `~matplotlib.font_manager.FontProperties`
    The font properties to search for. This can be either a
    `.FontProperties` object or a string defining a
    `fontconfig patterns`_.

fontext : {'ttf', 'afm'}, default: 'ttf'
    The extension of the font file:

    - 'ttf': TrueType and OpenType fonts (.ttf, .ttc, .otf)
    - 'afm': Adobe Font Metrics (.afm)

directory : str, optional
    If given, only search this directory and its subdirectories.

fallback_to_default : bool
    If True, will fall back to the default font family (usually
    "DejaVu Sans" or "Helvetica") if none of the families were found.

rebuild_if_missing : bool
    Whether to rebuild the font cache and search again if the first
    match appears to point to a nonexisting font (i.e., the font cache
    contains outdated entries).

Returns
-------
list[str]
    The paths of the fonts found.

Notes
-----
This is an extension/wrapper of the original findfont API, which only
returns a single font for given font properties. Instead, this API
returns a list of filepaths of multiple fonts which closely match the
given font properties.  Since this internally uses the original API,
there's no change to the logic of performing the nearest neighbor
search.  See `findfont` for more details.

### Function: _findfont_cached(self, prop, fontext, directory, fallback_to_default, rebuild_if_missing, rc_params)
