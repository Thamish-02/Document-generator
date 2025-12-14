## AI Summary

A file named latextools.py.


## Class: LaTeXTool

**Description:** An object to store configuration of the LaTeX tool.

### Function: latex_to_png(s, encode, backend, wrap, color, scale)

**Description:** Render a LaTeX string to PNG.

Parameters
----------
s : str
    The raw string containing valid inline LaTeX.
encode : bool, optional
    Should the PNG data base64 encoded to make it JSON'able.
backend : {matplotlib, dvipng}
    Backend for producing PNG data.
wrap : bool
    If true, Automatically wrap `s` as a LaTeX equation.
color : string
    Foreground color name among dvipsnames, e.g. 'Maroon' or on hex RGB
    format, e.g. '#AA20FA'.
scale : float
    Scale factor for the resulting PNG.
None is returned when the backend cannot be used.

### Function: latex_to_png_mpl(s, wrap, color, scale)

### Function: latex_to_png_dvipng(s, wrap, color, scale)

### Function: kpsewhich(filename)

**Description:** Invoke kpsewhich command with an argument `filename`.

### Function: genelatex(body, wrap)

**Description:** Generate LaTeX document for dvipng backend.

### Function: latex_to_html(s, alt)

**Description:** Render LaTeX to HTML with embedded PNG data using data URIs.

Parameters
----------
s : str
    The raw string containing valid inline LateX.
alt : str
    The alt text to use for the HTML.

### Function: _config_default(self)
