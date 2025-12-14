## AI Summary

A file named templating.py.


### Function: _default_template_ctx_processor()

**Description:** Default template context processor.  Injects `request`,
`session` and `g`.

## Class: Environment

**Description:** Works like a regular Jinja environment but has some additional
knowledge of how Flask's blueprint works so that it can prepend the
name of the blueprint to referenced templates if necessary.

## Class: DispatchingJinjaLoader

**Description:** A loader that looks for templates in the application and all
the blueprint folders.

### Function: _render(app, template, context)

### Function: render_template(template_name_or_list)

**Description:** Render a template by name with the given context.

:param template_name_or_list: The name of the template to render. If
    a list is given, the first name to exist will be rendered.
:param context: The variables to make available in the template.

### Function: render_template_string(source)

**Description:** Render a template from the given source string with the given
context.

:param source: The source code of the template to render.
:param context: The variables to make available in the template.

### Function: _stream(app, template, context)

### Function: stream_template(template_name_or_list)

**Description:** Render a template by name with the given context as a stream.
This returns an iterator of strings, which can be used as a
streaming response from a view.

:param template_name_or_list: The name of the template to render. If
    a list is given, the first name to exist will be rendered.
:param context: The variables to make available in the template.

.. versionadded:: 2.2

### Function: stream_template_string(source)

**Description:** Render a template from the given source string with the given
context as a stream. This returns an iterator of strings, which can
be used as a streaming response from a view.

:param source: The source code of the template to render.
:param context: The variables to make available in the template.

.. versionadded:: 2.2

### Function: __init__(self, app)

### Function: __init__(self, app)

### Function: get_source(self, environment, template)

### Function: _get_source_explained(self, environment, template)

### Function: _get_source_fast(self, environment, template)

### Function: _iter_loaders(self, template)

### Function: list_templates(self)

### Function: generate()
