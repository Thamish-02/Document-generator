## AI Summary

A file named application.py.


### Function: _preparse_for_subcommand(application_klass, argv)

**Description:** Preparse command line to look for subcommands.

### Function: _preparse_for_stopping_flags(application_klass, argv)

**Description:** Looks for 'help', 'version', and 'generate-config; commands
in command line. If found, raises the help and version of
current Application.

This is useful for traitlets applications that have to parse
the command line multiple times, but want to control when
when 'help' and 'version' is raised.

## Class: ExtensionAppJinjaMixin

**Description:** Use Jinja templates for HTML templates on top of an ExtensionApp.

## Class: JupyterServerExtensionException

**Description:** Exception class for raising for Server extensions errors.

## Class: ExtensionApp

**Description:** Base class for configurable Jupyter Server Extension Applications.

ExtensionApp subclasses can be initialized two ways:

- Extension is listed as a jpserver_extension, and ServerApp calls
  its load_jupyter_server_extension classmethod. This is the
  classic way of loading a server extension.

- Extension is launched directly by calling its `launch_instance`
  class method. This method can be set as a entry_point in
  the extensions setup.py.

### Function: _prepare_templates(self)

**Description:** Get templates defined in a subclass.

### Function: _default_open_browser(self)

### Function: config_file_paths(self)

**Description:** Look on the same path as our parent for config files

### Function: get_extension_package(cls)

**Description:** Get an extension package.

### Function: get_extension_point(cls)

**Description:** Get an extension point.

### Function: _default_url(self)

### Function: _default_serverapp(self)

### Function: _default_log_level(self)

### Function: _default_log_format(self)

**Description:** override default log format to include date & time

### Function: _default_static_url_prefix(self)

### Function: _config_file_name_default(self)

**Description:** The default config file name.

### Function: initialize_settings(self)

**Description:** Override this method to add handling of settings.

### Function: initialize_handlers(self)

**Description:** Override this method to append handlers to a Jupyter Server.

### Function: initialize_templates(self)

**Description:** Override this method to add handling of template files.

### Function: _prepare_config(self)

**Description:** Builds a Config object from the extension's traits and passes
the object to the webapp's settings as `<name>_config`.

### Function: _prepare_settings(self)

**Description:** Prepare the settings.

### Function: _prepare_handlers(self)

**Description:** Prepare the handlers.

### Function: _prepare_templates(self)

**Description:** Add templates to web app settings if extension has templates.

### Function: _jupyter_server_config(self)

**Description:** The jupyter server config.

### Function: _link_jupyter_server_extension(self, serverapp)

**Description:** Link the ExtensionApp to an initialized ServerApp.

The ServerApp is stored as an attribute and config
is exchanged between ServerApp and `self` in case
the command line contains traits for the ExtensionApp
or the ExtensionApp's config files have server
settings.

Note, the ServerApp has not initialized the Tornado
Web Application yet, so do not try to affect the
`web_app` attribute.

### Function: initialize(self)

**Description:** Initialize the extension app. The
corresponding server app and webapp should already
be initialized by this step.

- Appends Handlers to the ServerApp,
- Passes config and settings from ExtensionApp
  to the Tornado web application
- Points Tornado Webapp to templates and static assets.

### Function: start(self)

**Description:** Start the underlying Jupyter server.

Server should be started after extension is initialized.

### Function: current_activity(self)

**Description:** Return a list of activity happening in this extension.

### Function: stop(self)

**Description:** Stop the underlying Jupyter server.

### Function: _load_jupyter_server_extension(cls, serverapp)

**Description:** Initialize and configure this extension, then add the extension's
settings and handlers to the server's web application.

### Function: load_classic_server_extension(cls, serverapp)

**Description:** Enables extension to be loaded as classic Notebook (jupyter/notebook) extension.

### Function: make_serverapp(cls)

**Description:** Instantiate the ServerApp

Override to customize the ServerApp before it loads any configuration

### Function: initialize_server(cls, argv, load_other_extensions)

**Description:** Creates an instance of ServerApp and explicitly sets
this extension to enabled=True (i.e. superseding disabling
found in other config from files).

The `launch_instance` method uses this method to initialize
and start a server.

### Function: launch_instance(cls, argv)

**Description:** Launch the extension like an application. Initializes+configs a stock server
and appends the extension to the server. Then starts the server and routes to
extension's landing page.
