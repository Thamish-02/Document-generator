## AI Summary

A file named figureoptions.py.


### Function: figure_edit(axes, parent)

**Description:** Edit matplotlib figure options

### Function: convert_limits(lim, converter)

**Description:** Convert axis limits for correct input editors.

### Function: prepare_data(d, init)

**Description:** Prepare entry for FormLayout.

*d* is a mapping of shorthands to style names (a single style may
have multiple shorthands, in particular the shorthands `None`,
`"None"`, `"none"` and `""` are synonyms); *init* is one shorthand
of the initial style.

This function returns an list suitable for initializing a
FormLayout combobox, namely `[initial_name, (shorthand,
style_name), (shorthand, style_name), ...]`.

### Function: apply_callback(data)

**Description:** A callback to apply changes.
