## AI Summary

A file named federated_labextensions.py.


### Function: develop_labextension(path, symlink, overwrite, user, labextensions_dir, destination, logger, sys_prefix)

**Description:** Install a prebuilt extension for JupyterLab

Stages files and/or directories into the labextensions directory.
By default, this compares modification time, and only stages files that need updating.
If `overwrite` is specified, matching files are purged before proceeding.

Parameters
----------

path : path to file, directory, zip or tarball archive, or URL to install
    By default, the file will be installed with its base name, so '/path/to/foo'
    will install to 'labextensions/foo'. See the destination argument below to change this.
    Archives (zip or tarballs) will be extracted into the labextensions directory.
user : bool [default: False]
    Whether to install to the user's labextensions directory.
    Otherwise do a system-wide install (e.g. /usr/local/share/jupyter/labextensions).
overwrite : bool [default: False]
    If True, always install the files, regardless of what may already be installed.
symlink : bool [default: True]
    If True, create a symlink in labextensions, rather than copying files.
    Windows support for symlinks requires a permission bit which only admin users
    have by default, so don't rely on it.
labextensions_dir : str [optional]
    Specify absolute path of labextensions directory explicitly.
destination : str [optional]
    name the labextension is installed to.  For example, if destination is 'foo', then
    the source file will be installed to 'labextensions/foo', regardless of the source name.
logger : Jupyter logger [optional]
    Logger instance to use

### Function: develop_labextension_py(module, user, sys_prefix, overwrite, symlink, labextensions_dir, logger)

**Description:** Develop a labextension bundled in a Python package.

Returns a list of installed/updated directories.

See develop_labextension for parameter information.

### Function: build_labextension(path, logger, development, static_url, source_map, core_path)

**Description:** Build a labextension in the given path

### Function: watch_labextension(path, labextensions_path, logger, development, source_map, core_path)

**Description:** Watch a labextension in a given path

### Function: _ensure_builder(ext_path, core_path)

**Description:** Ensure that we can build the extension and return the builder script path

### Function: _should_copy(src, dest, logger)

**Description:** Should a file be copied, if it doesn't exist, or is newer?

Returns whether the file needs to be updated.

Parameters
----------

src : string
    A path that should exist from which to copy a file
src : string
    A path that might exist to which to copy a file
logger : Jupyter logger [optional]
    Logger instance to use

### Function: _maybe_copy(src, dest, logger)

**Description:** Copy a file if it needs updating.

Parameters
----------

src : string
    A path that should exist from which to copy a file
src : string
    A path that might exist to which to copy a file
logger : Jupyter logger [optional]
    Logger instance to use

### Function: _get_labextension_dir(user, sys_prefix, prefix, labextensions_dir)

**Description:** Return the labextension directory specified

Parameters
----------

user : bool [default: False]
    Get the user's .jupyter/labextensions directory
sys_prefix : bool [default: False]
    Get sys.prefix, i.e. ~/.envs/my-env/share/jupyter/labextensions
prefix : str [optional]
    Get custom prefix
labextensions_dir : str [optional]
    Get what you put in

### Function: _get_labextension_metadata(module)

**Description:** Get the list of labextension paths associated with a Python module.

Returns a tuple of (the module path,             [{
    'src': 'mockextension',
    'dest': '_mockdestination'
}])

Parameters
----------

module : str
    Importable Python module exposing the
    magic-named `_jupyter_labextension_paths` function
