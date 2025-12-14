## AI Summary

A file named embed.py.


### Function: _find_widget_refs_by_state(widget, state)

**Description:** Find references to other widgets in a widget's state

### Function: _get_recursive_state(widget, store, drop_defaults)

**Description:** Gets the embed state of a widget, and all other widgets it refers to as well

### Function: add_resolved_links(store, drop_defaults)

**Description:** Adds the state of any link models between two models in store

### Function: dependency_state(widgets, drop_defaults)

**Description:** Get the state of all widgets specified, and their dependencies.

This uses a simple dependency finder, including:
 - any widget directly referenced in the state of an included widget
 - any widget in a list/tuple attribute in the state of an included widget
 - any widget in a dict attribute in the state of an included widget
 - any jslink/jsdlink between two included widgets
What this alogorithm does not do:
 - Find widget references in nested list/dict structures
 - Find widget references in other types of attributes

Note that this searches the state of the widgets for references, so if
a widget reference is not included in the serialized state, it won't
be considered as a dependency.

Parameters
----------
widgets: single widget or list of widgets.
   This function will return the state of every widget mentioned
   and of all their dependencies.
drop_defaults: boolean
    Whether to drop default values from the widget states.

Returns
-------
A dictionary with the state of the widgets and any widget they
depend on.

### Function: embed_data(views, drop_defaults, state)

**Description:** Gets data for embedding.

Use this to get the raw data for embedding if you have special
formatting needs.

Parameters
----------
{views_attribute}
drop_defaults: boolean
    Whether to drop default values from the widget states.
state: dict or None (default)
    The state to include. When set to None, the state of all widgets
    know to the widget manager is included. Otherwise it uses the
    passed state directly. This allows for end users to include a
    smaller state, under the responsibility that this state is
    sufficient to reconstruct the embedded views.

Returns
-------
A dictionary with the following entries:
    manager_state: dict of the widget manager state data
    view_specs: a list of widget view specs

### Function: escape_script(s)

**Description:** Escape a string that will be the content of an HTML script tag.

We replace the opening bracket of <script, </script, and <!-- with the unicode
equivalent. This is inspired by the documentation for the script tag at
https://html.spec.whatwg.org/multipage/scripting.html#restrictions-for-contents-of-script-elements

We only replace these three cases so that most html or other content
involving `<` is readable.

### Function: embed_snippet(views, drop_defaults, state, indent, embed_url, requirejs, cors)

**Description:** Return a snippet that can be embedded in an HTML file.

Parameters
----------
{views_attribute}
{embed_kwargs}

Returns
-------
A unicode string with an HTML snippet containing several `<script>` tags.

### Function: embed_minimal_html(fp, views, title, template)

**Description:** Write a minimal HTML file with widget views embedded.

Parameters
----------
fp: filename or file-like object
    The file to write the HTML output to.
{views_attribute}
title: title of the html page.
template: Template in which to embed the widget state.
    This should be a Python string with placeholders
    `{{title}}` and `{{snippet}}`. The `{{snippet}}` placeholder
    will be replaced by all the widgets.
{embed_kwargs}
