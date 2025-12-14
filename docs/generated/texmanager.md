## AI Summary

A file named texmanager.py.


### Function: _usepackage_if_not_loaded(package)

**Description:** Output LaTeX code that loads a package (possibly with an option) if it
hasn't been loaded yet.

LaTeX cannot load twice a package with different options, so this helper
can be used to protect against users loading arbitrary packages/options in
their custom preamble.

## Class: TexManager

**Description:** Convert strings to dvi files using TeX, caching the results to a directory.

The cache directory is called ``tex.cache`` and is located in the directory
returned by `.get_cachedir`.

Repeated calls to this constructor always return the same instance.

### Function: __new__(cls)

### Function: _get_font_family_and_reduced(cls)

**Description:** Return the font family name and whether the font is reduced.

### Function: _get_font_preamble_and_command(cls)

### Function: get_basefile(cls, tex, fontsize, dpi)

**Description:** Return a filename based on a hash of the string, fontsize, and dpi.

### Function: get_font_preamble(cls)

**Description:** Return a string containing font configuration for the tex preamble.

### Function: get_custom_preamble(cls)

**Description:** Return a string containing user additions to the tex preamble.

### Function: _get_tex_source(cls, tex, fontsize)

**Description:** Return the complete TeX source for processing a TeX string.

### Function: make_tex(cls, tex, fontsize)

**Description:** Generate a tex file to render the tex string at a specific font size.

Return the file name.

### Function: _run_checked_subprocess(cls, command, tex)

### Function: make_dvi(cls, tex, fontsize)

**Description:** Generate a dvi file containing latex's layout of tex string.

Return the file name.

### Function: make_png(cls, tex, fontsize, dpi)

**Description:** Generate a png file containing latex's rendering of tex string.

Return the file name.

### Function: get_grey(cls, tex, fontsize, dpi)

**Description:** Return the alpha channel.

### Function: get_rgba(cls, tex, fontsize, dpi, rgb)

**Description:** Return latex's rendering of the tex string as an RGBA array.

Examples
--------
>>> texmanager = TexManager()
>>> s = r"\TeX\ is $\displaystyle\sum_n\frac{-e^{i\pi}}{2^n}$!"
>>> Z = texmanager.get_rgba(s, fontsize=12, dpi=80, rgb=(1, 0, 0))

### Function: get_text_width_height_descent(cls, tex, fontsize, renderer)

**Description:** Return width, height and descent of the text.
