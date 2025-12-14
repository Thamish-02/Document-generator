## AI Summary

A file named backend_managers.py.


## Class: ToolEvent

**Description:** Event for tool manipulation (add/remove).

## Class: ToolTriggerEvent

**Description:** Event to inform that a tool has been triggered.

## Class: ToolManagerMessageEvent

**Description:** Event carrying messages from toolmanager.

Messages usually get displayed to the user by the toolbar.

## Class: ToolManager

**Description:** Manager for actions triggered by user interactions (key press, toolbar
clicks, ...) on a Figure.

Attributes
----------
figure : `.Figure`
keypresslock : `~matplotlib.widgets.LockDraw`
    `.LockDraw` object to know if the `canvas` key_press_event is locked.
messagelock : `~matplotlib.widgets.LockDraw`
    `.LockDraw` object to know if the message is available to write.

### Function: __init__(self, name, sender, tool, data)

### Function: __init__(self, name, sender, tool, canvasevent, data)

### Function: __init__(self, name, sender, message)

### Function: __init__(self, figure)

### Function: canvas(self)

**Description:** Canvas managed by FigureManager.

### Function: figure(self)

**Description:** Figure that holds the canvas.

### Function: figure(self, figure)

### Function: set_figure(self, figure, update_tools)

**Description:** Bind the given figure to the tools.

Parameters
----------
figure : `.Figure`
update_tools : bool, default: True
    Force tools to update figure.

### Function: toolmanager_connect(self, s, func)

**Description:** Connect event with string *s* to *func*.

Parameters
----------
s : str
    The name of the event. The following events are recognized:

    - 'tool_message_event'
    - 'tool_removed_event'
    - 'tool_added_event'

    For every tool added a new event is created

    - 'tool_trigger_TOOLNAME', where TOOLNAME is the id of the tool.

func : callable
    Callback function for the toolmanager event with signature::

        def func(event: ToolEvent) -> Any

Returns
-------
cid
    The callback id for the connection. This can be used in
    `.toolmanager_disconnect`.

### Function: toolmanager_disconnect(self, cid)

**Description:** Disconnect callback id *cid*.

Example usage::

    cid = toolmanager.toolmanager_connect('tool_trigger_zoom', onpress)
    #...later
    toolmanager.toolmanager_disconnect(cid)

### Function: message_event(self, message, sender)

**Description:** Emit a `ToolManagerMessageEvent`.

### Function: active_toggle(self)

**Description:** Currently toggled tools.

### Function: get_tool_keymap(self, name)

**Description:** Return the keymap associated with the specified tool.

Parameters
----------
name : str
    Name of the Tool.

Returns
-------
list of str
    List of keys associated with the tool.

### Function: _remove_keys(self, name)

### Function: update_keymap(self, name, key)

**Description:** Set the keymap to associate with the specified tool.

Parameters
----------
name : str
    Name of the Tool.
key : str or list of str
    Keys to associate with the tool.

### Function: remove_tool(self, name)

**Description:** Remove tool named *name*.

Parameters
----------
name : str
    Name of the tool.

### Function: add_tool(self, name, tool)

**Description:** Add *tool* to `ToolManager`.

If successful, adds a new event ``tool_trigger_{name}`` where
``{name}`` is the *name* of the tool; the event is fired every time the
tool is triggered.

Parameters
----------
name : str
    Name of the tool, treated as the ID, has to be unique.
tool : type
    Class of the tool to be added.  A subclass will be used
    instead if one was registered for the current canvas class.
*args, **kwargs
    Passed to the *tool*'s constructor.

See Also
--------
matplotlib.backend_tools.ToolBase : The base class for tools.

### Function: _handle_toggle(self, tool, canvasevent, data)

**Description:** Toggle tools, need to untoggle prior to using other Toggle tool.
Called from trigger_tool.

Parameters
----------
tool : `.ToolBase`
canvasevent : Event
    Original Canvas event or None.
data : object
    Extra data to pass to the tool when triggering.

### Function: trigger_tool(self, name, sender, canvasevent, data)

**Description:** Trigger a tool and emit the ``tool_trigger_{name}`` event.

Parameters
----------
name : str
    Name of the tool.
sender : object
    Object that wishes to trigger the tool.
canvasevent : Event
    Original Canvas event or None.
data : object
    Extra data to pass to the tool when triggering.

### Function: _key_press(self, event)

### Function: tools(self)

**Description:** A dict mapping tool name -> controlled tool.

### Function: get_tool(self, name, warn)

**Description:** Return the tool object with the given name.

For convenience, this passes tool objects through.

Parameters
----------
name : str or `.ToolBase`
    Name of the tool, or the tool itself.
warn : bool, default: True
    Whether a warning should be emitted it no tool with the given name
    exists.

Returns
-------
`.ToolBase` or None
    The tool or None if no tool with the given name exists.
