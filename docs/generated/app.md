## AI Summary

A file named app.py.


## Class: NotebookBaseHandler

**Description:** The base notebook API handler.

## Class: TreeHandler

**Description:** A tree page handler.

## Class: ConsoleHandler

**Description:** A console page handler.

## Class: TerminalHandler

**Description:** A terminal page handler.

## Class: FileHandler

**Description:** A file page handler.

## Class: NotebookHandler

**Description:** A notebook page handler.

## Class: CustomCssHandler

**Description:** A custom CSS handler.

## Class: JupyterNotebookApp

**Description:** The notebook server extension app.

### Function: custom_css(self)

### Function: get_page_config(self)

**Description:** Get the page config.

### Function: get(self, path)

**Description:** Get the console page.

### Function: get(self, path)

**Description:** Get the terminal page.

### Function: get(self, path)

**Description:** Get the file page.

### Function: get(self)

**Description:** Get the custom css file.

### Function: _default_static_dir(self)

### Function: _default_templates_dir(self)

### Function: _default_app_settings_dir(self)

### Function: _default_schemas_dir(self)

### Function: _default_themes_dir(self)

### Function: _default_user_settings_dir(self)

### Function: _default_workspaces_dir(self)

### Function: _prepare_templates(self)

### Function: server_extension_is_enabled(self, extension)

**Description:** Check if server extension is enabled.

### Function: initialize_handlers(self)

**Description:** Initialize handlers.

### Function: initialize(self, argv)

**Description:** Subclass because the ExtensionApp.initialize() method does not take arguments
