## AI Summary

A file named logger.py.


## Class: SchemaNotRegistered

**Description:** A warning to raise when an event is given to the logger
but its schema has not be registered with the EventLogger

## Class: ModifierError

**Description:** An exception to raise when a modifier does not
show the proper signature.

## Class: CoreMetadataError

**Description:** An exception raised when event core metadata is not valid.

## Class: ListenerError

**Description:** An exception to raise when a listener does not
show the proper signature.

## Class: EventLogger

**Description:** An Event logger for emitting structured events.

Event schemas must be registered with the
EventLogger using the `register_schema` or
`register_schema_file` methods. Every schema
will be validated against Jupyter Event's metaschema.

### Function: _default_schemas(self)

### Function: __init__(self)

**Description:** Initialize the logger.

### Function: _load_config(self, cfg, section_names, traits)

**Description:** Load EventLogger traits from a Config object, patching the
handlers trait in the Config object to avoid deepcopy errors.

### Function: register_event_schema(self, schema)

**Description:** Register this schema with the schema registry.

Get this registered schema using the EventLogger.schema.get() method.

### Function: register_handler(self, handler)

**Description:** Register a new logging handler to the Event Logger.

All outgoing messages will be formatted as a JSON string.

### Function: remove_handler(self, handler)

**Description:** Remove a logging handler from the logger and list of handlers.

### Function: add_modifier(self)

**Description:** Add a modifier (callable) to a registered event.

Parameters
----------
modifier: Callable
    A callable function/method that executes when the named event occurs.
    This method enforces a string signature for modifiers:

        (schema_id: str, data: dict) -> dict:

### Function: remove_modifier(self)

**Description:** Remove a modifier from an event or all events.

Parameters
----------
schema_id: str
    If given, remove this modifier only for a specific event type.
modifier: Callable[[str, dict], dict]

    The modifier to remove.

### Function: add_listener(self)

**Description:** Add a listener (callable) to a registered event.

Parameters
----------
modified: bool
    If True (default), listens to the data after it has been mutated/modified
    by the list of modifiers.
schema_id: str
    $id of the schema
listener: Callable
    A callable function/method that executes when the named event occurs.

### Function: remove_listener(self)

**Description:** Remove a listener from an event or all events.

Parameters
----------
schema_id: str
    If given, remove this modifier only for a specific event type.

listener: Callable[[EventLogger, str, dict], dict]
    The modifier to remove.

### Function: emit(self)

**Description:** Record given event with schema has occurred.

Parameters
----------
schema_id: str
    $id of the schema
data: dict
    The event to record
timestamp_override: datetime, optional
    Optionally override the event timestamp. By default it is set to the current timestamp.

Returns
-------
dict
    The recorded event data

### Function: get_handlers()

### Function: _handle_message_field(record)

**Description:** Python's logger always emits the "message" field with
the value as "null" unless it's present in the schema/data.
Message happens to be a common field for event logs,
so special case it here and only emit it if "message"
is found the in the schema's property list.

### Function: _listener_task_done(task)

### Function: _listener_task_done(task)
