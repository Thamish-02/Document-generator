## AI Summary

A file named backend_tools.py.


## Class: Cursors

**Description:** Backend-independent cursor types.

### Function: _register_tool_class(canvas_cls, tool_cls)

**Description:** Decorator registering *tool_cls* as a tool class for *canvas_cls*.

### Function: _find_tool_class(canvas_cls, tool_cls)

**Description:** Find a subclass of *tool_cls* registered for *canvas_cls*.

## Class: ToolBase

**Description:** Base tool class.

A base tool, only implements `trigger` method or no method at all.
The tool is instantiated by `matplotlib.backend_managers.ToolManager`.

## Class: ToolToggleBase

**Description:** Toggleable tool.

Every time it is triggered, it switches between enable and disable.

Parameters
----------
``*args``
    Variable length argument to be used by the Tool.
``**kwargs``
    `toggled` if present and True, sets the initial state of the Tool
    Arbitrary keyword arguments to be consumed by the Tool

## Class: ToolSetCursor

**Description:** Change to the current cursor while inaxes.

This tool, keeps track of all `ToolToggleBase` derived tools, and updates
the cursor when a tool gets triggered.

## Class: ToolCursorPosition

**Description:** Send message with the current pointer position.

This tool runs in the background reporting the position of the cursor.

## Class: RubberbandBase

**Description:** Draw and remove a rubberband.

## Class: ToolQuit

**Description:** Tool to call the figure manager destroy method.

## Class: ToolQuitAll

**Description:** Tool to call the figure manager destroy method.

## Class: ToolGrid

**Description:** Tool to toggle the major grids of the figure.

## Class: ToolMinorGrid

**Description:** Tool to toggle the major and minor grids of the figure.

## Class: ToolFullScreen

**Description:** Tool to toggle full screen.

## Class: AxisScaleBase

**Description:** Base Tool to toggle between linear and logarithmic.

## Class: ToolYScale

**Description:** Tool to toggle between linear and logarithmic scales on the Y axis.

## Class: ToolXScale

**Description:** Tool to toggle between linear and logarithmic scales on the X axis.

## Class: ToolViewsPositions

**Description:** Auxiliary Tool to handle changes in views and positions.

Runs in the background and should get used by all the tools that
need to access the figure's history of views and positions, e.g.

* `ToolZoom`
* `ToolPan`
* `ToolHome`
* `ToolBack`
* `ToolForward`

## Class: ViewsPositionsBase

**Description:** Base class for `ToolHome`, `ToolBack` and `ToolForward`.

## Class: ToolHome

**Description:** Restore the original view limits.

## Class: ToolBack

**Description:** Move back up the view limits stack.

## Class: ToolForward

**Description:** Move forward in the view lim stack.

## Class: ConfigureSubplotsBase

**Description:** Base tool for the configuration of subplots.

## Class: SaveFigureBase

**Description:** Base tool for figure saving.

## Class: ZoomPanBase

**Description:** Base class for `ToolZoom` and `ToolPan`.

## Class: ToolZoom

**Description:** A Tool for zooming using a rectangle selector.

## Class: ToolPan

**Description:** Pan Axes with left mouse, zoom with right.

## Class: ToolHelpBase

## Class: ToolCopyToClipboardBase

**Description:** Tool to copy the figure to the clipboard.

### Function: add_tools_to_manager(toolmanager, tools)

**Description:** Add multiple tools to a `.ToolManager`.

Parameters
----------
toolmanager : `.backend_managers.ToolManager`
    Manager to which the tools are added.
tools : {str: class_like}, optional
    The tools to add in a {name: tool} dict, see
    `.backend_managers.ToolManager.add_tool` for more info.

### Function: add_tools_to_container(container, tools)

**Description:** Add multiple tools to the container.

Parameters
----------
container : Container
    `.backend_bases.ToolContainerBase` object that will get the tools
    added.
tools : list, optional
    List in the form ``[[group1, [tool1, tool2 ...]], [group2, [...]]]``
    where the tools ``[tool1, tool2, ...]`` will display in group1.
    See `.backend_bases.ToolContainerBase.add_tool` for details.

### Function: __init__(self, toolmanager, name)

### Function: set_figure(self, figure)

### Function: _make_classic_style_pseudo_toolbar(self)

**Description:** Return a placeholder object with a single `canvas` attribute.

This is useful to reuse the implementations of tools already provided
by the classic Toolbars.

### Function: trigger(self, sender, event, data)

**Description:** Called when this tool gets used.

This method is called by `.ToolManager.trigger_tool`.

Parameters
----------
event : `.Event`
    The canvas event that caused this tool to be called.
sender : object
    Object that requested the tool to be triggered.
data : object
    Extra data.

### Function: __init__(self)

### Function: trigger(self, sender, event, data)

**Description:** Calls `enable` or `disable` based on `toggled` value.

### Function: enable(self, event)

**Description:** Enable the toggle tool.

`trigger` calls this method when `toggled` is False.

### Function: disable(self, event)

**Description:** Disable the toggle tool.

`trigger` call this method when `toggled` is True.

This can happen in different circumstances.

* Click on the toolbar tool button.
* Call to `matplotlib.backend_managers.ToolManager.trigger_tool`.
* Another `ToolToggleBase` derived tool is triggered
  (from the same `.ToolManager`).

### Function: toggled(self)

**Description:** State of the toggled tool.

### Function: set_figure(self, figure)

### Function: __init__(self)

### Function: set_figure(self, figure)

### Function: _add_tool_cbk(self, event)

**Description:** Process every newly added tool.

### Function: _tool_trigger_cbk(self, event)

### Function: _set_cursor_cbk(self, event)

### Function: __init__(self)

### Function: set_figure(self, figure)

### Function: send_message(self, event)

**Description:** Call `matplotlib.backend_managers.ToolManager.message_event`.

### Function: trigger(self, sender, event, data)

**Description:** Call `draw_rubberband` or `remove_rubberband` based on data.

### Function: draw_rubberband(self)

**Description:** Draw rubberband.

This method must get implemented per backend.

### Function: remove_rubberband(self)

**Description:** Remove rubberband.

This method should get implemented per backend.

### Function: trigger(self, sender, event, data)

### Function: trigger(self, sender, event, data)

### Function: trigger(self, sender, event, data)

### Function: trigger(self, sender, event, data)

### Function: trigger(self, sender, event, data)

### Function: trigger(self, sender, event, data)

### Function: enable(self, event)

### Function: disable(self, event)

### Function: set_scale(self, ax, scale)

### Function: set_scale(self, ax, scale)

### Function: __init__(self)

### Function: add_figure(self, figure)

**Description:** Add the current figure to the stack of views and positions.

### Function: clear(self, figure)

**Description:** Reset the Axes stack.

### Function: update_view(self)

**Description:** Update the view limits and position for each Axes from the current
stack position. If any Axes are present in the figure that aren't in
the current stack position, use the home view limits for those Axes and
don't update *any* positions.

### Function: push_current(self, figure)

**Description:** Push the current view limits and position onto their respective stacks.

### Function: _axes_pos(self, ax)

**Description:** Return the original and modified positions for the specified Axes.

Parameters
----------
ax : matplotlib.axes.Axes
    The `.Axes` to get the positions for.

Returns
-------
original_position, modified_position
    A tuple of the original and modified positions.

### Function: update_home_views(self, figure)

**Description:** Make sure that ``self.home_views`` has an entry for all Axes present
in the figure.

### Function: home(self)

**Description:** Recall the first view and position from the stack.

### Function: back(self)

**Description:** Back one step in the stack of views and positions.

### Function: forward(self)

**Description:** Forward one step in the stack of views and positions.

### Function: trigger(self, sender, event, data)

### Function: __init__(self)

### Function: enable(self, event)

**Description:** Connect press/release events and lock the canvas.

### Function: disable(self, event)

**Description:** Release the canvas and disconnect press/release events.

### Function: trigger(self, sender, event, data)

### Function: scroll_zoom(self, event)

### Function: __init__(self)

### Function: _cancel_action(self)

### Function: _press(self, event)

**Description:** Callback for mouse button presses in zoom-to-rectangle mode.

### Function: _switch_on_zoom_mode(self, event)

### Function: _switch_off_zoom_mode(self, event)

### Function: _mouse_move(self, event)

**Description:** Callback for mouse moves in zoom-to-rectangle mode.

### Function: _release(self, event)

**Description:** Callback for mouse button releases in zoom-to-rectangle mode.

### Function: __init__(self)

### Function: _cancel_action(self)

### Function: _press(self, event)

### Function: _release(self, event)

### Function: _mouse_move(self, event)

### Function: format_shortcut(key_sequence)

**Description:** Convert a shortcut string from the notation used in rc config to the
standard notation for displaying shortcuts, e.g. 'ctrl+a' -> 'Ctrl+A'.

### Function: _format_tool_keymap(self, name)

### Function: _get_help_entries(self)

### Function: _get_help_text(self)

### Function: _get_help_html(self)

### Function: trigger(self)
