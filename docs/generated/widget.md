## AI Summary

A file named widget.py.


### Function: envset(name, default)

**Description:** Return True if the given environment variable is turned on, otherwise False
If the environment variable is set, True will be returned if it is assigned to a value
other than 'no', 'n', 'false', 'off', '0', or '0.0' (case insensitive).
If the environment variable is not set, the default value is returned.

### Function: _widget_to_json(x, obj)

### Function: _json_to_widget(x, obj)

### Function: _put_buffers(state, buffer_paths, buffers)

**Description:** The inverse of _remove_buffers, except here we modify the existing dict/lists.
Modifying should be fine, since this is used when state comes from the wire.

### Function: _separate_buffers(substate, path, buffer_paths, buffers)

**Description:** For internal, see _remove_buffers

### Function: _remove_buffers(state)

**Description:** Return (state_without_buffers, buffer_paths, buffers) for binary message parts

A binary message part is a memoryview, bytearray, or python 3 bytes object.

As an example:
>>> state = {'plain': [0, 'text'], 'x': {'ar': memoryview(ar1)}, 'y': {'shape': (10,10), 'data': memoryview(ar2)}}
>>> _remove_buffers(state)
({'plain': [0, 'text']}, {'x': {}, 'y': {'shape': (10, 10)}}, [['x', 'ar'], ['y', 'data']],
 [<memory at 0x107ffec48>, <memory at 0x107ffed08>])

### Function: _buffer_list_equal(a, b)

**Description:** Compare two lists of buffers for equality.

Used to decide whether two sequences of buffers (memoryviews,
bytearrays, or python 3 bytes) differ, such that a sync is needed.

Returns True if equal, False if unequal

## Class: LoggingHasTraits

**Description:** A parent class for HasTraits that log.
Subclasses have a log trait, and the default behavior
is to get the logger from the currently running Application.

## Class: CallbackDispatcher

**Description:** A structure for registering and running callbacks

### Function: _show_traceback(method)

**Description:** decorator for showing tracebacks

## Class: WidgetRegistry

### Function: register(widget)

**Description:** A decorator registering a widget class in the widget registry.

## Class: _staticproperty

## Class: Widget

### Function: _log_default(self)

### Function: __call__(self)

**Description:** Call all of the registered callbacks.

### Function: register_callback(self, callback, remove)

**Description:** (Un)Register a callback

Parameters
----------
callback: method handle
    Method to be registered or unregistered.
remove=False: bool
    Whether to unregister the callback.

### Function: m(self)

### Function: __init__(self)

### Function: register(self, model_module, model_module_version_range, model_name, view_module, view_module_version_range, view_name, klass)

**Description:** Register a value

### Function: get(self, model_module, model_module_version, model_name, view_module, view_module_version, view_name)

**Description:** Get a value

### Function: items(self)

### Function: __init__(self, fget)

### Function: __get__(self, owner_self, owner_cls)

### Function: widgets()

### Function: _active_widgets()

### Function: _widget_types()

### Function: widget_types()

### Function: close_all(cls)

### Function: on_widget_constructed(callback)

**Description:** Registers a callback to be called when a widget is constructed.

The callback must have the following signature:
callback(widget)

### Function: _call_widget_constructed(widget)

**Description:** Static method, called when a widget is constructed.

### Function: handle_control_comm_opened(cls, comm, msg)

**Description:** Class method, called when the comm-open message on the
"jupyter.widget.control" comm channel is received

### Function: _handle_control_comm_msg(cls, msg)

### Function: handle_comm_opened(comm, msg)

**Description:** Static method, called when a widget is constructed.

### Function: get_manager_state(drop_defaults, widgets)

**Description:** Returns the full state for a widget manager for embedding

:param drop_defaults: when True, it will not include default value
:param widgets: list with widgets to include in the state (or all widgets when None)
:return:

### Function: _get_embed_state(self, drop_defaults)

### Function: get_view_spec(self)

### Function: _default_keys(self)

### Function: __init__(self)

**Description:** Public constructor

### Function: __copy__(self)

### Function: __deepcopy__(self, memo)

### Function: __del__(self)

**Description:** Object disposal

### Function: open(self)

**Description:** Open a comm to the frontend if one isn't already open.

### Function: _comm_changed(self, change)

**Description:** Called when the comm is changed.

### Function: model_id(self)

**Description:** Gets the model id of this widget.

If a Comm doesn't exist yet, a Comm will be created automagically.

### Function: close(self)

**Description:** Close method.

Closes the underlying comm.
When the comm is closed, all of the widget views are automatically
removed from the front-end.

### Function: send_state(self, key)

**Description:** Sends the widget state, or a piece of it, to the front-end, if it exists.

Parameters
----------
key : unicode, or iterable (optional)
    A single property's name or iterable of property names to sync with the front-end.

### Function: get_state(self, key, drop_defaults)

**Description:** Gets the widget state, or a piece of it.

Parameters
----------
key : unicode or iterable (optional)
    A single property's name or iterable of property names to get.

Returns
-------
state : dict of states
metadata : dict
    metadata for each field: {key: metadata}

### Function: _is_numpy(self, x)

### Function: _compare(self, a, b)

### Function: set_state(self, sync_data)

**Description:** Called when a state is received from the front-end.

### Function: send(self, content, buffers)

**Description:** Sends a custom msg to the widget model in the front-end.

Parameters
----------
content : dict
    Content of the message to send.
buffers : list of binary buffers
    Binary buffers to send with message

### Function: on_msg(self, callback, remove)

**Description:** (Un)Register a custom msg receive callback.

Parameters
----------
callback: callable
    callback will be passed three arguments when a message arrives::

        callback(widget, content, buffers)

remove: bool
    True if the callback should be unregistered.

### Function: add_traits(self)

**Description:** Dynamically add trait attributes to the Widget.

### Function: notify_change(self, change)

**Description:** Called when a property has changed.

### Function: __repr__(self)

### Function: _lock_property(self)

**Description:** Lock a property-value pair.

The value should be the JSON state of the property.

NOTE: This, in addition to the single lock for all state changes, is
flawed.  In the future we may want to look into buffering state changes
back to the front-end.

### Function: hold_sync(self)

**Description:** Hold syncing any state until the outermost context manager exits

### Function: _should_send_property(self, key, value)

**Description:** Check the property lock (property_lock)

### Function: _handle_msg(self, msg)

**Description:** Called when a msg is received from the front-end

### Function: _handle_custom_msg(self, content, buffers)

**Description:** Called when a custom msg is received.

### Function: _trait_to_json(x, self)

**Description:** Convert a trait value to json.

### Function: _trait_from_json(x, self)

**Description:** Convert json values to objects.

### Function: _repr_mimebundle_(self)

### Function: _send(self, msg, buffers)

**Description:** Sends a message to the model in the front-end.

### Function: _repr_keys(self)

### Function: _gen_repr_from_keys(self, keys)
