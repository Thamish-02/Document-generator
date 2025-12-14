## AI Summary

A file named nbconvertapp.py.


## Class: DottedOrNone

**Description:** A string holding a valid dotted object name in Python, such as A.b3._c
Also allows for None type.

## Class: NbConvertApp

**Description:** Application used to convert from notebook file type (``*.ipynb``)

## Class: DejavuApp

**Description:** A deja vu app.

### Function: validate(self, obj, value)

**Description:** Validate an input.

### Function: _log_level_default(self)

### Function: _classes_default(self)

### Function: _writer_class_changed(self, change)

### Function: _postprocessor_class_changed(self, change)

### Function: initialize(self, argv)

**Description:** Initialize application, notebooks, writer, and postprocessor

### Function: init_syspath(self)

**Description:** Add the cwd to the sys.path ($PYTHONPATH)

### Function: init_notebooks(self)

**Description:** Construct the list of notebooks.

If notebooks are passed on the command-line,
they override (rather than add) notebooks specified in config files.
Glob each notebook to replace notebook patterns with filenames.

### Function: init_writer(self)

**Description:** Initialize the writer (which is stateless)

### Function: init_postprocessor(self)

**Description:** Initialize the postprocessor (which is stateless)

### Function: start(self)

**Description:** Run start after initialization process has completed

### Function: _notebook_filename_to_name(self, notebook_filename)

**Description:** Returns the notebook name from the notebook filename by
applying `output_base` pattern and stripping extension

### Function: init_single_notebook_resources(self, notebook_filename)

**Description:** Step 1: Initialize resources

This initializes the resources dictionary for a single notebook.

Returns
-------
dict
    resources dictionary for a single notebook that MUST include the following keys:
        - config_dir: the location of the Jupyter config directory
        - unique_key: the notebook name
        - output_files_dir: a directory where output files (not
          including the notebook itself) should be saved

### Function: export_single_notebook(self, notebook_filename, resources, input_buffer)

**Description:** Step 2: Export the notebook

Exports the notebook to a particular format according to the specified
exporter. This function returns the output and (possibly modified)
resources from the exporter.

Parameters
----------
notebook_filename : str
    name of notebook file.
resources : dict
input_buffer :
    readable file-like object returning unicode.
    if not None, notebook_filename is ignored

Returns
-------
output
dict
    resources (possibly modified)

### Function: write_single_notebook(self, output, resources)

**Description:** Step 3: Write the notebook to file

This writes output from the exporter to file using the specified writer.
It returns the results from the writer.

Parameters
----------
output :
resources : dict
    resources for a single notebook including name, config directory
    and directory to save output

Returns
-------
file
    results from the specified writer output of exporter

### Function: postprocess_single_notebook(self, write_results)

**Description:** Step 4: Post-process the written file

Only used if a postprocessor has been specified. After the
converted notebook is written to a file in Step 3, this post-processes
the notebook.

### Function: convert_single_notebook(self, notebook_filename, input_buffer)

**Description:** Convert a single notebook.

Performs the following steps:

    1. Initialize notebook resources
    2. Export the notebook to a particular format
    3. Write the exported notebook to file
    4. (Maybe) postprocess the written file

Parameters
----------
notebook_filename : str
input_buffer :
    If input_buffer is not None, conversion is done and the buffer is
    used as source into a file basenamed by the notebook_filename
    argument.

### Function: convert_notebooks(self)

**Description:** Convert the notebooks in the self.notebooks traitlet

### Function: document_flag_help(self)

**Description:** Return a string containing descriptions of all the flags.

### Function: document_alias_help(self)

**Description:** Return a string containing all of the aliases

### Function: document_config_options(self)

**Description:** Provides a much improves version of the configuration documentation by
breaking the configuration options into app, exporter, writer,
preprocessor, postprocessor, and other sections.

### Function: initialize(self, argv)

**Description:** Initialize the app.

### Function: _default_export_format(self)
