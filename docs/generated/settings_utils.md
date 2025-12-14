## AI Summary

A file named settings_utils.py.


### Function: _get_schema(schemas_dir, schema_name, overrides, labextensions_path)

**Description:** Returns a dict containing a parsed and validated JSON schema.

### Function: _get_user_settings(settings_dir, schema_name, schema)

**Description:** Returns a dictionary containing the raw user settings, the parsed user
settings, a validation warning for a schema, and file times.

### Function: _get_version(schemas_dir, schema_name)

**Description:** Returns the package version for a given schema or 'N/A' if not found.

### Function: _list_settings(schemas_dir, settings_dir, overrides, extension, labextensions_path, translator, ids_only)

**Description:** Returns a tuple containing:
 - the list of plugins, schemas, and their settings,
   respecting any defaults that may have been overridden if `ids_only=False`,
   otherwise a list of dict containing only the ids of plugins.
 - the list of warnings that were generated when
   validating the user overrides against the schemas.

### Function: _override(schema_name, schema, overrides)

**Description:** Override default values in the schema if necessary.

### Function: _path(root_dir, schema_name, make_dirs, extension)

**Description:** Returns the local file system path for a schema name in the given root
directory. This function can be used to filed user overrides in addition to
schema files. If the `make_dirs` flag is set to `True` it will create the
parent directory for the calculated path if it does not exist.

### Function: _get_overrides(app_settings_dir)

**Description:** Get overrides settings from `app_settings_dir`.

The ordering of paths is:
- {app_settings_dir}/overrides.d/*.{json,json5} (many, namespaced by package)
- {app_settings_dir}/overrides.{json,json5} (singleton, owned by the user)

### Function: get_settings(app_settings_dir, schemas_dir, settings_dir, schema_name, overrides, labextensions_path, translator, ids_only)

**Description:** Get settings.

Parameters
----------
app_settings_dir:
    Path to applications settings.
schemas_dir: str
    Path to schemas.
settings_dir:
    Path to settings.
schema_name str, optional
    Schema name. Default is "".
overrides: dict, optional
    Settings overrides. If not provided, the overrides will be loaded
    from the `app_settings_dir`. Default is None.
labextensions_path: list, optional
    List of paths to federated labextensions containing their own schema files.
translator: Callable[[Dict], Dict] or None, optional
    Translate a schema. It requires the schema dictionary and returns its translation

Returns
-------
tuple
    The first item is a dictionary with a list of setting if no `schema_name`
    was provided (only the ids if `ids_only=True`), otherwise it is a dictionary
    with id, raw, scheme, settings and version keys.
    The second item is a list of warnings. Warnings will either be a list of
    i) strings with the warning messages or ii) `None`.

### Function: save_settings(schemas_dir, settings_dir, schema_name, raw_settings, overrides, labextensions_path)

**Description:** Save ``raw_settings`` settings for ``schema_name``.

Parameters
----------
schemas_dir: str
    Path to schemas.
settings_dir: str
    Path to settings.
schema_name str
    Schema name.
raw_settings: str
    Raw serialized settings dictionary
overrides: dict
    Settings overrides.
labextensions_path: list, optional
    List of paths to federated labextensions containing their own schema files.

## Class: SchemaHandler

**Description:** Base handler for handler requiring access to settings.

### Function: initialize(self, app_settings_dir, schemas_dir, settings_dir, labextensions_path, overrides)

**Description:** Initialize the handler.

### Function: get_current_locale(self)

**Description:** Get the current locale as specified in the translation-extension settings.

Returns
-------
str
    The current locale string.

Notes
-----
If the locale setting is not available or not valid, it will default to jupyterlab_server.translation_utils.DEFAULT_LOCALE.
