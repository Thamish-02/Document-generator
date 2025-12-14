## AI Summary

A file named pypi.py.


## Class: ProxiedTransport

## Class: PyPIExtensionManager

**Description:** Extension manager using pip as package manager and PyPi.org as packages source.

### Function: set_proxy(self, host, port, headers)

### Function: make_connection(self, host)

### Function: __init__(self, app_options, ext_options, parent)

### Function: metadata(self)

**Description:** Extension manager metadata.

### Function: get_normalized_name(self, extension)

**Description:** Normalize extension name.

Extension have multiple parts, npm package, Python package,...
Sub-classes may override this method to ensure the name of
an extension from the service provider and the local installed
listing is matching.

Args:
    extension: The extension metadata
Returns:
    The normalized name

### Function: _observe_package_metadata_cache_size(self, change)

### Function: _normalize_name(self, name)

**Description:** Normalize extension name.

Remove `@` from npm scope and replace `/` and `_` by `-`.

Args:
    name: Extension name
Returns:
    Normalized name
