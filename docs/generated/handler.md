## AI Summary

A file named handler.py.


## Class: ExtensionHandlerJinjaMixin

**Description:** Mixin class for ExtensionApp handlers that use jinja templating for
template rendering.

## Class: ExtensionHandlerMixin

**Description:** Base class for Jupyter server extension handlers.

Subclasses can serve static files behind a namespaced
endpoint: "<base_url>/static/<name>/"

This allows multiple extensions to serve static files under
their own namespace and avoid intercepting requests for
other extensions.

### Function: get_template(self, name)

**Description:** Return the jinja template object for a given name

### Function: initialize(self, name)

### Function: extensionapp(self)

### Function: serverapp(self)

### Function: log(self)

### Function: config(self)

### Function: server_config(self)

### Function: base_url(self)

### Function: render_template(self, name)

**Description:** Override render template to handle static_paths

If render_template is called with a template from the base environment
(e.g. default error pages)
make sure our extension-specific static_url is _not_ used.

### Function: static_url_prefix(self)

### Function: static_path(self)

### Function: static_url(self, path, include_host)

**Description:** Returns a static URL for the given relative static file path.
This method requires you set the ``{name}_static_path``
setting in your extension (which specifies the root directory
of your static files).
This method returns a versioned url (by default appending
``?v=<signature>``), which allows the static files to be
cached indefinitely.  This can be disabled by passing
``include_version=False`` (in the default implementation;
other static file implementations are not required to support
this, but they may support other options).
By default this method returns URLs relative to the current
host, but if ``include_host`` is true the URL returned will be
absolute.  If this handler has an ``include_host`` attribute,
that value will be used as the default for all `static_url`
calls that do not pass ``include_host`` as a keyword argument.
