## AI Summary

A file named script.py.


## Class: ScriptExporter

**Description:** A script exporter.

### Function: _template_file_default(self)

### Function: _template_name_default(self)

### Function: _get_language_exporter(self, lang_name)

**Description:** Find an exporter for the language name from notebook metadata.

Uses the nbconvert.exporters.script group of entry points.
Returns None if no exporter is found.

### Function: from_notebook_node(self, nb, resources)

**Description:** Convert from notebook node.
