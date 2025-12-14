## AI Summary

A file named backend_nbagg.py.


### Function: connection_info()

**Description:** Return a string showing the figure and connection status for the backend.

This is intended as a diagnostic tool, and not for general use.

## Class: NavigationIPy

## Class: FigureManagerNbAgg

## Class: FigureCanvasNbAgg

## Class: CommSocket

**Description:** Manages the Comm connection between IPython and the browser (client).

Comms are 2 way, with the CommSocket being able to publish a message
via the send_json method, and handle a message with on_message. On the
JS side figure.send_message and figure.ws.onmessage do the sending and
receiving respectively.

## Class: _BackendNbAgg

### Function: __init__(self, canvas, num)

### Function: create_with_canvas(cls, canvas_class, figure, num)

### Function: display_js(self)

### Function: show(self)

### Function: reshow(self)

**Description:** A special method to re-show the figure in the notebook.

### Function: connected(self)

### Function: get_javascript(cls, stream)

### Function: _create_comm(self)

### Function: destroy(self)

### Function: clearup_closed(self)

**Description:** Clear up any closed Comms.

### Function: remove_comm(self, comm_id)

### Function: __init__(self, manager)

### Function: is_open(self)

### Function: on_close(self)

### Function: send_json(self, content)

### Function: send_binary(self, blob)

### Function: on_message(self, message)

### Function: destroy(event)

### Function: _on_close(close_message)
