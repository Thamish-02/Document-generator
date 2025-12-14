## AI Summary

A file named backend_pgf.py.


### Function: _get_preamble()

**Description:** Prepare a LaTeX preamble based on the rcParams configuration.

### Function: _tex_escape(text)

**Description:** Do some necessary and/or useful substitutions for texts to be included in
LaTeX documents.

### Function: _writeln(fh, line)

### Function: _escape_and_apply_props(s, prop)

**Description:** Generate a TeX string that renders string *s* with font properties *prop*,
also applying any required escapes to *s*.

### Function: _metadata_to_str(key, value)

**Description:** Convert metadata key/value to a form that hyperref accepts.

### Function: make_pdf_to_png_converter()

**Description:** Return a function that converts a pdf file to a png file.

## Class: LatexError

## Class: LatexManager

**Description:** The LatexManager opens an instance of the LaTeX application for
determining the metrics of text elements. The LaTeX environment can be
modified by setting fonts and/or a custom preamble in `.rcParams`.

### Function: _get_image_inclusion_command()

## Class: RendererPgf

## Class: FigureCanvasPgf

## Class: _BackendPgf

## Class: PdfPages

**Description:** A multi-page PDF file using the pgf backend

Examples
--------
>>> import matplotlib.pyplot as plt
>>> # Initialize:
>>> with PdfPages('foo.pdf') as pdf:
...     # As many times as you like, create a figure fig and save it:
...     fig = plt.figure()
...     pdf.savefig(fig)
...     # When no figure is specified the current figure is saved
...     pdf.savefig()

### Function: __init__(self, message, latex_output)

### Function: __str__(self)

### Function: _build_latex_header()

### Function: _get_cached_or_new(cls)

**Description:** Return the previous LatexManager if the header and tex system did not
change, or a new instance otherwise.

### Function: _get_cached_or_new_impl(cls, header)

### Function: _stdin_writeln(self, s)

### Function: _expect(self, s)

### Function: _expect_prompt(self)

### Function: __init__(self)

### Function: _setup_latex_process(self)

### Function: get_width_height_descent(self, text, prop)

**Description:** Get the width, total height, and descent (in TeX points) for a text
typeset by the current LaTeX environment.

### Function: _get_box_metrics(self, tex)

**Description:** Get the width, total height and descent (in TeX points) for a TeX
command's output in the current LaTeX environment.

### Function: __init__(self, figure, fh)

**Description:** Create a new PGF renderer that translates any drawing instruction
into text commands to be interpreted in a latex pgfpicture environment.

Attributes
----------
figure : `~matplotlib.figure.Figure`
    Matplotlib figure to initialize height, width and dpi from.
fh : file-like
    File handle for the output of the drawing commands.

### Function: draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace)

### Function: draw_path(self, gc, path, transform, rgbFace)

### Function: _print_pgf_clip(self, gc)

### Function: _print_pgf_path_styles(self, gc, rgbFace)

### Function: _print_pgf_path(self, gc, path, transform, rgbFace)

### Function: _pgf_path_draw(self, stroke, fill)

### Function: option_scale_image(self)

### Function: option_image_nocomposite(self)

### Function: draw_image(self, gc, x, y, im, transform)

### Function: draw_tex(self, gc, x, y, s, prop, angle)

### Function: draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)

### Function: get_text_width_height_descent(self, s, prop, ismath)

### Function: flipy(self)

### Function: get_canvas_width_height(self)

### Function: points_to_pixels(self, points)

### Function: get_default_filetype(self)

### Function: _print_pgf_to_fh(self, fh)

### Function: print_pgf(self, fname_or_fh)

**Description:** Output pgf macros for drawing the figure so it can be included and
rendered in latex documents.

### Function: print_pdf(self, fname_or_fh)

**Description:** Use LaTeX to compile a pgf generated figure to pdf.

### Function: print_png(self, fname_or_fh)

**Description:** Use LaTeX to compile a pgf figure to pdf and convert it to png.

### Function: get_renderer(self)

### Function: draw(self)

### Function: __init__(self, filename)

**Description:** Create a new PdfPages object.

Parameters
----------
filename : str or path-like
    Plots using `PdfPages.savefig` will be written to a file at this
    location. Any older file with the same name is overwritten.

metadata : dict, optional
    Information dictionary object (see PDF reference section 10.2.1
    'Document Information Dictionary'), e.g.:
    ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

    The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
    'Creator', 'Producer', 'CreationDate', 'ModDate', and
    'Trapped'. Values have been predefined for 'Creator', 'Producer'
    and 'CreationDate'. They can be removed by setting them to `None`.

    Note that some versions of LaTeX engines may ignore the 'Producer'
    key and set it to themselves.

### Function: _write_header(self, width_inches, height_inches)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: close(self)

**Description:** Finalize this object, running LaTeX in a temporary directory
and moving the final pdf file to *filename*.

### Function: _run_latex(self)

### Function: savefig(self, figure)

**Description:** Save a `.Figure` to this file as a new page.

Any other keyword arguments are passed to `~.Figure.savefig`.

Parameters
----------
figure : `.Figure` or int, default: the active figure
    The figure, or index of the figure, that is saved to the file.

### Function: get_pagecount(self)

**Description:** Return the current number of pages in the multipage pdf file.

### Function: finalize_latex(latex)
