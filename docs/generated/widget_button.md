## AI Summary

A file named widget_button.py.


## Class: ButtonStyle

**Description:** Button style widget.

## Class: Button

**Description:** Button widget.

This widget has an `on_click` method that allows you to listen for the
user clicking on the button.  The click event itself is stateless.

Parameters
----------
description: str
   description displayed on the button
icon: str
   font-awesome icon names, without the 'fa-' prefix
disabled: bool
   whether user interaction is enabled

### Function: __init__(self)

### Function: _validate_icon(self, proposal)

**Description:** Strip 'fa-' if necessary'

### Function: on_click(self, callback, remove)

**Description:** Register a callback to execute when the button is clicked.

The callback will be called with one argument, the clicked button
widget instance.

Parameters
----------
remove: bool (optional)
    Set to true to remove the callback from the list of callbacks.

### Function: click(self)

**Description:** Programmatically trigger a click event.

This will call the callbacks registered to the clicked button
widget instance.

### Function: _handle_button_msg(self, _, content, buffers)

**Description:** Handle a msg from the front-end.

Parameters
----------
content: dict
    Content of the msg.
