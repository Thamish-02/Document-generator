## AI Summary

A file named workspaces_handler.py.


### Function: _list_workspaces(directory, prefix)

**Description:** Return the list of workspaces in a given directory beginning with the
given prefix.

### Function: _load_with_file_times(workspace_path)

**Description:** Load workspace JSON from disk, overwriting the `created` and `last_modified`
metadata with current file stat information

### Function: slugify(raw, base, sign, max_length)

**Description:** Use the common superset of raw and base values to build a slug shorter
than max_length. By default, base value is an empty string.
Convert spaces to hyphens. Remove characters that aren't alphanumerics
underscores, or hyphens. Convert to lowercase. Strip leading and trailing
whitespace.
Add an optional short signature suffix to prevent collisions.
Modified from Django utils:
https://github.com/django/django/blob/master/django/utils/text.py

## Class: WorkspacesManager

**Description:** A manager for workspaces.

## Class: WorkspacesHandler

**Description:** A workspaces API handler.

### Function: __init__(self, path)

**Description:** Initialize a workspaces manager with content in ``path``.

### Function: delete(self, space_name)

**Description:** Remove a workspace ``space_name``.

### Function: list_workspaces(self)

**Description:** List all available workspaces.

### Function: load(self, space_name)

**Description:** Load the workspace ``space_name``.

### Function: save(self, space_name, raw)

**Description:** Save the ``raw`` data as workspace ``space_name``.

### Function: initialize(self, name, manager)

**Description:** Initialize the handler.

### Function: delete(self, space_name)

**Description:** Remove a workspace

### Function: put(self, space_name)

**Description:** Update workspace data
