## AI Summary

A file named plot_directive.py.


### Function: _option_boolean(arg)

### Function: _option_context(arg)

### Function: _option_format(arg)

### Function: mark_plot_labels(app, document)

**Description:** To make plots referenceable, we need to move the reference from the
"htmlonly" (or "latexonly") node to the actual figure node itself.

## Class: PlotDirective

**Description:** The ``.. plot::`` directive, as documented in the module's docstring.

### Function: _copy_css_file(app, exc)

### Function: setup(app)

### Function: contains_doctest(text)

### Function: _split_code_at_show(text, function_name)

**Description:** Split code at plt.show().

## Class: ImageFile

### Function: out_of_date(original, derived, includes)

**Description:** Return whether *derived* is out-of-date relative to *original* or any of
the RST files included in it using the RST include directive (*includes*).
*derived* and *original* are full paths, and *includes* is optionally a
list of full paths which may have been included in the *original*.

## Class: PlotError

### Function: _run_code(code, code_path, ns, function_name)

**Description:** Import a Python module from a path, and run the function given by
name, if function_name is not None.

### Function: clear_state(plot_rcparams, close)

### Function: get_plot_formats(config)

### Function: _parse_srcset(entries)

**Description:** Parse srcset for multiples...

### Function: render_figures(code, code_path, output_dir, output_base, context, function_name, config, context_reset, close_figs, code_includes)

**Description:** Run a pyplot script and save the images in *output_dir*.

Save the images under *output_dir* with file names derived from
*output_base*

### Function: run(arguments, content, options, state_machine, state, lineno)

### Function: run(self)

**Description:** Run the plot directive.

### Function: __init__(self, basename, dirname)

### Function: filename(self, format)

### Function: filenames(self)

### Function: out_of_date_one(original, derived_mtime)
