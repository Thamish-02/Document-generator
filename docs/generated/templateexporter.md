## AI Summary

A file named templateexporter.py.


### Function: recursive_update(target, new)

**Description:** Recursively update one dictionary using another.
None values will delete their keys.

### Function: deprecated(msg)

**Description:** Emit a deprecation warning.

## Class: ExtensionTolerantLoader

**Description:** A template loader which optionally adds a given extension when searching.

Constructor takes two arguments: *loader* is another Jinja loader instance
to wrap. *extension* is the extension, which will be added to the template
name if finding the template without it fails. This should include the dot,
e.g. '.tpl'.

## Class: TemplateExporter

**Description:** Exports notebooks into other file formats.  Uses Jinja 2 templating engine
to output new formats.  Inherit from this class if you are creating a new
template type along with new filters/preprocessors.  If the filters/
preprocessors provided by default suffice, there is no need to inherit from
this class.  Instead, override the template_file and file_extension
traits via a config file.

Filters available by default for templates:

{filters}

### Function: __init__(self, loader, extension)

**Description:** Initialize the loader.

### Function: get_source(self, environment, template)

**Description:** Get the source for a template.

### Function: list_templates(self)

**Description:** List available templates.

### Function: _invalidate_template_cache(self, change)

### Function: template(self)

### Function: _invalidate_environment_cache(self, change)

### Function: environment(self)

### Function: default_config(self)

### Function: _template_name_validate(self, change)

### Function: _template_file_changed(self, change)

### Function: _template_file_default(self)

### Function: _raw_template_changed(self, change)

### Function: _default_extra_template_basedirs(self)

### Function: _template_extension_default(self)

### Function: _raw_mimetypes_default(self)

### Function: __init__(self, config)

**Description:** Public constructor

Parameters
----------
config : config
    User configuration instance.
extra_loaders : list[of Jinja Loaders]
    ordered list of Jinja loader to find templates. Will be tried in order
    before the default FileSystem ones.
template_file : str (optional, kw arg)
    Template to use when exporting.

### Function: _load_template(self)

**Description:** Load the Jinja template object from the template file

This is triggered by various trait changes that would change the template.

### Function: from_filename(self, filename, resources)

**Description:** Convert a notebook from a filename.

### Function: from_file(self, file_stream, resources)

**Description:** Convert a notebook from a file.

### Function: from_notebook_node(self, nb, resources)

**Description:** Convert a notebook from a notebook node instance.

Parameters
----------
nb : :class:`~nbformat.NotebookNode`
    Notebook node
resources : dict
    Additional resources that can be accessed read/write by
    preprocessors and filters.

### Function: _register_filter(self, environ, name, jinja_filter)

**Description:** Register a filter.
A filter is a function that accepts and acts on one string.
The filters are accessible within the Jinja templating engine.

Parameters
----------
name : str
    name to give the filter in the Jinja engine
filter : filter

### Function: register_filter(self, name, jinja_filter)

**Description:** Register a filter.
A filter is a function that accepts and acts on one string.
The filters are accessible within the Jinja templating engine.

Parameters
----------
name : str
    name to give the filter in the Jinja engine
filter : filter

### Function: default_filters(self)

**Description:** Override in subclasses to provide extra filters.

This should return an iterable of 2-tuples: (name, class-or-function).
You should call the method on the parent class and include the filters
it provides.

If a name is repeated, the last filter provided wins. Filters from
user-supplied config win over filters provided by classes.

### Function: _create_environment(self)

**Description:** Create the Jinja templating environment.

### Function: _init_preprocessors(self)

### Function: _get_conf(self)

### Function: _template_paths(self, prune, root_dirs)

### Function: get_compatibility_base_template_conf(cls, name)

**Description:** Get the base template config.

### Function: get_template_names(self)

**Description:** Finds a list of template names where each successive template name is the base template

### Function: get_prefix_root_dirs(self)

**Description:** Get the prefix root dirs.

### Function: _init_resources(self, resources)
