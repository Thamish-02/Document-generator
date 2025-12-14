## AI Summary

A file named serverextension.py.


### Function: _get_config_dir(user, sys_prefix)

**Description:** Get the location of config files for the current context

Returns the string to the environment

Parameters
----------
user : bool [default: False]
    Get the user's .jupyter config directory
sys_prefix : bool [default: False]
    Get sys.prefix, i.e. ~/.envs/my-env/etc/jupyter

### Function: _get_extmanager_for_context(write_dir, user, sys_prefix)

**Description:** Get an extension manager pointing at the current context

Returns the path to the current context and an ExtensionManager object.

Parameters
----------
write_dir : str [default: 'jupyter_server_config.d']
    Name of config directory to write extension config.
user : bool [default: False]
    Get the user's .jupyter config directory
sys_prefix : bool [default: False]
    Get sys.prefix, i.e. ~/.envs/my-env/etc/jupyter

## Class: ArgumentConflict

## Class: BaseExtensionApp

**Description:** Base extension installer app

### Function: toggle_server_extension_python(import_name, enabled, parent, user, sys_prefix)

**Description:** Toggle the boolean setting for a given server extension
in a Jupyter config file.

## Class: ToggleServerExtensionApp

**Description:** A base class for enabling/disabling extensions

## Class: EnableServerExtensionApp

**Description:** An App that enables (and validates) Server Extensions

## Class: DisableServerExtensionApp

**Description:** An App that disables Server Extensions

## Class: ListServerExtensionsApp

**Description:** An App that lists (and validates) Server Extensions

## Class: ServerExtensionApp

**Description:** Root level server extension app

### Function: _log_format_default(self)

**Description:** A default format for messages

### Function: config_dir(self)

### Function: toggle_server_extension(self, import_name)

**Description:** Change the status of a named server extension.

Uses the value of `self._toggle_value`.

Parameters
---------

import_name : str
    Importable Python module (dotted-notation) exposing the magic-named
    `load_jupyter_server_extension` function

### Function: start(self)

**Description:** Perform the App's actions as configured

### Function: list_server_extensions(self)

**Description:** List all enabled and disabled server extensions, by config path

Enabled extensions are validated, potentially generating warnings.

### Function: start(self)

**Description:** Perform the App's actions as configured

### Function: start(self)

**Description:** Perform the App's actions as configured
