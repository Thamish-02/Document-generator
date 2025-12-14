## AI Summary

A file named widget_string.py.


## Class: _StringStyle

**Description:** Text input style widget.

## Class: LabelStyle

**Description:** Label style widget.

## Class: TextStyle

**Description:** Text input style widget.

## Class: HTMLStyle

**Description:** HTML style widget.

## Class: HTMLMathStyle

**Description:** HTML with math style widget.

## Class: _String

**Description:** Base class used to create widgets that represent a string.

## Class: HTML

**Description:** Renders the string `value` as HTML.

## Class: HTMLMath

**Description:** Renders the string `value` as HTML, and render mathematics.

## Class: Label

**Description:** Label widget.

It also renders math inside the string `value` as Latex (requires $ $ or
$$ $$ and similar latex tags).

## Class: Textarea

**Description:** Multiline text area widget.

## Class: Text

**Description:** Single line textbox widget.

## Class: Password

**Description:** Single line textbox widget.

## Class: Combobox

**Description:** Single line textbox widget with a dropdown and autocompletion.
    

### Function: __init__(self, value)

### Function: __init__(self)

### Function: _handle_string_msg(self, _, content, buffers)

**Description:** Handle a msg from the front-end.

Parameters
----------
content: dict
    Content of the msg.

### Function: on_submit(self, callback, remove)

**Description:** (Un)Register a callback to handle text submission.

Triggered when the user clicks enter.

Parameters
----------
callback: callable
    Will be called with exactly one argument: the Widget instance
remove: bool (optional)
    Whether to unregister the callback

### Function: _repr_keys(self)
