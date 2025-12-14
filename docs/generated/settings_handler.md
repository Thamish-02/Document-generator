## AI Summary

A file named settings_handler.py.


## Class: SettingsHandler

**Description:** A settings API handler.

### Function: initialize(self, name, app_settings_dir, schemas_dir, settings_dir, labextensions_path, overrides)

**Description:** Initialize the handler.

### Function: get(self, schema_name)

**Description:** Get setting(s)

Parameters
----------
schema_name: str
    The id of a unique schema to send, added to the URL

## NOTES:
    An optional argument `ids_only=true` can be provided in the URL to get only the
    ids of the schemas instead of the content.

### Function: put(self, schema_name)

**Description:** Update a setting
