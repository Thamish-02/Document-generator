## AI Summary

A file named pdf.py.


## Class: LatexFailed

**Description:** Exception for failed latex run

Captured latex output is in error.output.

### Function: prepend_to_env_search_path(varname, value, envdict)

**Description:** Add value to the environment variable varname in envdict

e.g. prepend_to_env_search_path('BIBINPUTS', '/home/sally/foo', os.environ)

## Class: PDFExporter

**Description:** Writer designed to write to PDF files.

This inherits from `LatexExporter`. It creates a LaTeX file in
a temporary directory using the template machinery, and then runs LaTeX
to create a pdf.

### Function: __init__(self, output)

**Description:** Initialize the error.

### Function: __unicode__(self)

**Description:** Unicode representation.

### Function: __str__(self)

**Description:** String representation.

### Function: _file_extension_default(self)

### Function: _template_extension_default(self)

### Function: run_command(self, command_list, filename, count, log_function, raise_on_failure)

**Description:** Run command_list count times.

Parameters
----------
command_list : list
    A list of args to provide to Popen. Each element of this
    list will be interpolated with the filename to convert.
filename : unicode
    The name of the file to convert.
count : int
    How many times to run the command.
raise_on_failure: Exception class (default None)
    If provided, will raise the given exception for if an instead of
    returning False on command failure.

Returns
-------
success : bool
    A boolean indicating if the command was successful (True)
    or failed (False).

### Function: run_latex(self, filename, raise_on_failure)

**Description:** Run xelatex self.latex_count times.

### Function: run_bib(self, filename, raise_on_failure)

**Description:** Run bibtex one time.

### Function: from_notebook_node(self, nb, resources)

**Description:** Convert from notebook node.

### Function: log_error(command, out)

### Function: log_error(command, out)
