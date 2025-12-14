## AI Summary

A file named html.py.


### Function: find_lab_theme(theme_name)

**Description:** Find a JupyterLab theme location by name.

Parameters
----------
theme_name : str
    The name of the labextension theme you want to find.

Raises
------
ValueError
    If the theme was not found, or if it was not specific enough.

Returns
-------
theme_name: str
    Full theme name (with scope, if any)
labextension_path : Path
    The path to the found labextension on the system.

## Class: HTMLExporter

**Description:** Exports a basic HTML document.  This exporter assists with the export of
HTML.  Inherit from it if you are writing your own HTML template and need
custom preprocessors/filters.  If you don't need custom preprocessors/
filters, just change the 'template_file' config option.

### Function: _file_extension_default(self)

### Function: _template_name_default(self)

### Function: default_config(self)

### Function: _valid_language_code(self, proposal)

### Function: markdown2html(self, context, source)

**Description:** Markdown to HTML filter respecting the anchor_link_text setting

### Function: default_filters(self)

**Description:** Get the default filters.

### Function: from_notebook_node(self, nb, resources)

**Description:** Convert from notebook node.

### Function: _init_resources(self, resources)

### Function: resources_include_css(name)

### Function: resources_include_lab_theme(name)

### Function: resources_include_js(name, module)

**Description:** Get the resources include JS for a name. If module=True, import as ES module

### Function: resources_include_url(name)

**Description:** Get the resources include url for a name.
