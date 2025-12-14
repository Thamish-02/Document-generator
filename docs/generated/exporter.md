## AI Summary

A file named exporter.py.


## Class: ResourcesDict

**Description:** A default dict for resources.

## Class: FilenameExtension

**Description:** A trait for filename extensions.

## Class: Exporter

**Description:** Class containing methods that sequentially run a list of preprocessors on a
NotebookNode object and then return the modified NotebookNode object and
accompanying resources dict.

### Function: __missing__(self, key)

**Description:** Handle missing value.

### Function: validate(self, obj, value)

**Description:** Validate the file name.

### Function: __init__(self, config)

**Description:** Public constructor

Parameters
----------
config : ``traitlets.config.Config``
    User configuration instance.
`**kw`
    Additional keyword arguments passed to parent __init__

### Function: default_config(self)

### Function: from_notebook_node(self, nb, resources)

**Description:** Convert a notebook from a notebook node instance.

Parameters
----------
nb : :class:`~nbformat.NotebookNode`
    Notebook node (dict-like with attr-access)
resources : dict
    Additional resources that can be accessed read/write by
    preprocessors and filters.
`**kw`
    Ignored

### Function: from_filename(self, filename, resources)

**Description:** Convert a notebook from a notebook file.

Parameters
----------
filename : str
    Full filename of the notebook file to open and convert.
resources : dict
    Additional resources that can be accessed read/write by
    preprocessors and filters.
`**kw`
    Ignored

### Function: from_file(self, file_stream, resources)

**Description:** Convert a notebook from a notebook file.

Parameters
----------
file_stream : file-like object
    Notebook file-like object to convert.
resources : dict
    Additional resources that can be accessed read/write by
    preprocessors and filters.
`**kw`
    Ignored

### Function: register_preprocessor(self, preprocessor, enabled)

**Description:** Register a preprocessor.
Preprocessors are classes that act upon the notebook before it is
passed into the Jinja templating engine. Preprocessors are also
capable of passing additional information to the Jinja
templating engine.

Parameters
----------
preprocessor : `nbconvert.preprocessors.Preprocessor`
    A dotted module name, a type, or an instance
enabled : bool
    Mark the preprocessor as enabled

### Function: _init_preprocessors(self)

**Description:** Register all of the preprocessors needed for this exporter, disabled
unless specified explicitly.

### Function: _init_resources(self, resources)

### Function: _validate_preprocessor(self, nbc, preprocessor)

### Function: _preprocess(self, nb, resources)

**Description:** Preprocess the notebook before passing it into the Jinja engine.
To preprocess the notebook is to successively apply all the
enabled preprocessors. Output from each preprocessor is passed
along to the next one.

Parameters
----------
nb : notebook node
    notebook that is being exported.
resources : a dict of additional resources that
    can be accessed read/write by preprocessors
