## AI Summary

A file named widgets.py.


### Function: get_ax()

**Description:** Create a plot and return its Axes.

### Function: noop()

### Function: mock_event(ax, button, xdata, ydata, key, step)

**Description:** Create a mock event that can stand in for `.Event` and its subclasses.

This event is intended to be used in tests where it can be passed into
event handling functions.

Parameters
----------
ax : `~matplotlib.axes.Axes`
    The Axes the event will be in.
xdata : float
    x coord of mouse in data coords.
ydata : float
    y coord of mouse in data coords.
button : None or `MouseButton` or {'up', 'down'}
    The mouse button pressed in this event (see also `.MouseEvent`).
key : None or str
    The key pressed when the mouse event triggered (see also `.KeyEvent`).
step : int
    Number of scroll steps (positive for 'up', negative for 'down').

Returns
-------
event
    A `.Event`\-like Mock instance.

### Function: do_event(tool, etype, button, xdata, ydata, key, step)

**Description:** Trigger an event on the given tool.

Parameters
----------
tool : matplotlib.widgets.AxesWidget
etype : str
    The event to trigger.
xdata : float
    x coord of mouse in data coords.
ydata : float
    y coord of mouse in data coords.
button : None or `MouseButton` or {'up', 'down'}
    The mouse button pressed in this event (see also `.MouseEvent`).
key : None or str
    The key pressed when the mouse event triggered (see also `.KeyEvent`).
step : int
    Number of scroll steps (positive for 'up', negative for 'down').

### Function: click_and_drag(tool, start, end, key)

**Description:** Helper to simulate a mouse drag operation.

Parameters
----------
tool : `~matplotlib.widgets.Widget`
start : [float, float]
    Starting point in data coordinates.
end : [float, float]
    End point in data coordinates.
key : None or str
     An optional key that is pressed during the whole operation
     (see also `.KeyEvent`).
