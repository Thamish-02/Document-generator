## AI Summary

A file named coreconfig.py.


### Function: pjoin()

**Description:** Join paths to create a real path.

### Function: _get_default_core_data()

**Description:** Get the data for the app template.

### Function: _is_lab_package(name)

**Description:** Whether a package name is in the lab namespace

### Function: _only_nonlab(collection)

**Description:** Filter a dict/sequence to remove all lab packages

This is useful to take the default values of e.g. singletons and filter
away the '@jupyterlab/' namespace packages, but leave any others (e.g.
lumino and react).

## Class: CoreConfig

**Description:** An object representing a core config.

This enables custom lab application to override some parts of the core
configuration of the build system.

### Function: __init__(self)

### Function: add(self, name, semver, extension, mime_extension)

**Description:** Remove an extension/singleton.

If neither extension or mimeExtension is True (the default)
the package is added as a singleton dependency.

name: string
    The npm package name
semver: string
    The semver range for the package
extension: bool
    Whether the package is an extension
mime_extension: bool
    Whether the package is a MIME extension

### Function: remove(self, name)

**Description:** Remove a package/extension.

name: string
    The npm package name

### Function: clear_packages(self, lab_only)

**Description:** Clear the packages/extensions.

### Function: extensions(self)

**Description:** A dict mapping all extension names to their semver

### Function: mime_extensions(self)

**Description:** A dict mapping all MIME extension names to their semver

### Function: singletons(self)

**Description:** A dict mapping all singleton names to their semver

### Function: static_dir(self)

### Function: static_dir(self, static_dir)
