## AI Summary

A file named messaging.py.


## Class: JsonIOError

**Description:** Indicates that a read or write operation on JsonIOStream has failed.

## Class: NoMoreMessages

**Description:** Indicates that there are no more messages that can be read from or written
to a stream.

## Class: JsonIOStream

**Description:** Implements a JSON value stream over two byte streams (input and output).

Each value is encoded as a DAP packet, with metadata headers and a JSON payload.

## Class: MessageDict

**Description:** A specialized dict that is used for JSON message payloads - Request.arguments,
Response.body, and Event.body.

For all members that normally throw KeyError when a requested key is missing, this
dict raises InvalidMessageError instead. Thus, a message handler can skip checks
for missing properties, and just work directly with the payload on the assumption
that it is valid according to the protocol specification; if anything is missing,
it will be reported automatically in the proper manner.

If the value for the requested key is itself a dict, it is returned as is, and not
automatically converted to MessageDict. Thus, to enable convenient chaining - e.g.
d["a"]["b"]["c"] - the dict must consistently use MessageDict instances rather than
vanilla dicts for all its values, recursively. This is guaranteed for the payload
of all freshly received messages (unless and until it is mutated), but there is no
such guarantee for outgoing messages.

### Function: _payload(value)

**Description:** JSON validator for message payload.

If that value is missing or null, it is treated as if it were {}.

## Class: Message

**Description:** Represents a fully parsed incoming or outgoing message.

https://microsoft.github.io/debug-adapter-protocol/specification#protocolmessage

## Class: Event

**Description:** Represents an incoming event.

https://microsoft.github.io/debug-adapter-protocol/specification#event

It is guaranteed that body is a MessageDict associated with this Event, and so
are all the nested dicts in it. If "body" was missing or null in JSON, body is
an empty dict.

To handle the event, JsonMessageChannel tries to find a handler for this event in
JsonMessageChannel.handlers. Given event="X", if handlers.X_event exists, then it
is the specific handler for this event. Otherwise, handlers.event must exist, and
it is the generic handler for this event. A missing handler is a fatal error.

No further incoming messages are processed until the handler returns, except for
responses to requests that have wait_for_response() invoked on them.

To report failure to handle the event, the handler must raise an instance of
MessageHandlingError that applies_to() the Event object it was handling. Any such
failure is logged, after which the message loop moves on to the next message.

Helper methods Message.isnt_valid() and Message.cant_handle() can be used to raise
the appropriate exception type that applies_to() the Event object.

## Class: Request

**Description:** Represents an incoming or an outgoing request.

Incoming requests are represented directly by instances of this class.

Outgoing requests are represented by instances of OutgoingRequest, which provides
additional functionality to handle responses.

For incoming requests, it is guaranteed that arguments is a MessageDict associated
with this Request, and so are all the nested dicts in it. If "arguments" was missing
or null in JSON, arguments is an empty dict.

To handle the request, JsonMessageChannel tries to find a handler for this request
in JsonMessageChannel.handlers. Given command="X", if handlers.X_request exists,
then it is the specific handler for this request. Otherwise, handlers.request must
exist, and it is the generic handler for this request. A missing handler is a fatal
error.

The handler is then invoked with the Request object as its sole argument.

If the handler itself invokes respond() on the Request at any point, then it must
not return any value.

Otherwise, if the handler returns NO_RESPONSE, no response to the request is sent.
It must be sent manually at some later point via respond().

Otherwise, a response to the request is sent with the returned value as the body.

To fail the request, the handler can return an instance of MessageHandlingError,
or respond() with one, or raise one such that it applies_to() the Request object
being handled.

Helper methods Message.isnt_valid() and Message.cant_handle() can be used to raise
the appropriate exception type that applies_to() the Request object.

## Class: OutgoingRequest

**Description:** Represents an outgoing request, for which it is possible to wait for a
response to be received, and register a response handler.

## Class: Response

**Description:** Represents an incoming or an outgoing response to a Request.

https://microsoft.github.io/debug-adapter-protocol/specification#response

error_message corresponds to "message" in JSON, and is renamed for clarity.

If success is False, body is None. Otherwise, it is a MessageDict associated
with this Response, and so are all the nested dicts in it. If "body" was missing
or null in JSON, body is an empty dict.

If this is a response to an outgoing request, it will be handled by the handler
registered via self.request.on_response(), if any.

Regardless of whether there is such a handler, OutgoingRequest.wait_for_response()
can also be used to retrieve and handle the response. If there is a handler, it is
executed before wait_for_response() returns.

No further incoming messages are processed until the handler returns, except for
responses to requests that have wait_for_response() invoked on them.

To report failure to handle the event, the handler must raise an instance of
MessageHandlingError that applies_to() the Response object it was handling. Any
such failure is logged, after which the message loop moves on to the next message.

Helper methods Message.isnt_valid() and Message.cant_handle() can be used to raise
the appropriate exception type that applies_to() the Response object.

## Class: Disconnect

**Description:** A dummy message used to represent disconnect. It's always the last message
received from any channel.

## Class: MessageHandlingError

**Description:** Indicates that a message couldn't be handled for some reason.

If the reason is a contract violation - i.e. the message that was handled did not
conform to the protocol specification - InvalidMessageError, which is a subclass,
should be used instead.

If any message handler raises an exception not derived from this class, it will
escape the message loop unhandled, and terminate the process.

If any message handler raises this exception, but applies_to(message) is False, it
is treated as if it was a generic exception, as desribed above. Thus, if a request
handler issues another request of its own, and that one fails, the failure is not
silently propagated. However, a request that is delegated via Request.delegate()
will also propagate failures back automatically. For manual propagation, catch the
exception, and call exc.propagate().

If any event handler raises this exception, and applies_to(event) is True, the
exception is silently swallowed by the message loop.

If any request handler raises this exception, and applies_to(request) is True, the
exception is silently swallowed by the message loop, and a failure response is sent
with "message" set to str(reason).

Note that, while errors are not logged when they're swallowed by the message loop,
by that time they have already been logged by their __init__ (when instantiated).

## Class: InvalidMessageError

**Description:** Indicates that an incoming message did not follow the protocol specification -
for example, it was missing properties that are required, or the message itself
is not allowed in the current state.

Raised by MessageDict in lieu of KeyError for missing keys.

## Class: JsonMessageChannel

**Description:** Implements a JSON message channel on top of a raw JSON message stream, with
support for DAP requests, responses, and events.

The channel can be locked for exclusive use via the with-statement::

    with channel:
        channel.send_request(...)
        # No interleaving messages can be sent here from other threads.
        channel.send_event(...)

## Class: MessageHandlers

**Description:** A simple delegating message handlers object for use with JsonMessageChannel.
For every argument provided, the object gets an attribute with the corresponding
name and value.

### Function: __init__(self)

### Function: __init__(self)

### Function: from_stdio(cls, name)

**Description:** Creates a new instance that receives messages from sys.stdin, and sends
them to sys.stdout.

### Function: from_process(cls, process, name)

**Description:** Creates a new instance that receives messages from process.stdin, and sends
them to process.stdout.

### Function: from_socket(cls, sock, name)

**Description:** Creates a new instance that sends and receives messages over a socket.

### Function: __init__(self, reader, writer, name, cleanup)

**Description:** Creates a new JsonIOStream.

        reader must be a BytesIO-like object, from which incoming messages will be
        read by read_json().

        writer must be a BytesIO-like object, into which outgoing messages will be
        written by write_json().

        cleanup must be a callable; it will be invoked without arguments when the
        stream is closed.

        reader.readline() must treat "
" as the line terminator, and must leave ""
        as is - it must not replace "
" with "
" automatically, as TextIO does.
        

### Function: close(self)

**Description:** Closes the stream, the reader, and the writer.

### Function: _log_message(self, dir, data, logger)

### Function: _read_line(self, reader)

### Function: read_json(self, decoder)

**Description:** Read a single JSON value from reader.

Returns JSON value as parsed by decoder.decode(), or raises NoMoreMessages
if there are no more values to be read.

### Function: write_json(self, value, encoder)

**Description:** Write a single JSON value into writer.

Value is written as encoded by encoder.encode().

### Function: __repr__(self)

### Function: __init__(self, message, items)

### Function: __repr__(self)

### Function: __call__(self, key, validate, optional)

**Description:** Like get(), but with validation.

The item is first retrieved as if with self.get(key, default=()) - the default
value is () rather than None, so that JSON nulls are distinguishable from
missing properties.

If optional=True, and the value is (), it's returned as is. Otherwise, the
item is validated by invoking validate(item) on it.

If validate=False, it's treated as if it were (lambda x: x) - i.e. any value
is considered valid, and is returned unchanged. If validate is a type or a
tuple, it's treated as json.of_type(validate). Otherwise, if validate is not
callable(), it's treated as json.default(validate).

If validate() returns successfully, the item is substituted with the value
it returns - thus, the validator can e.g. replace () with a suitable default
value for the property.

If validate() raises TypeError or ValueError, raises InvalidMessageError with
the same text that applies_to(self.messages).

See debugpy.common.json for reusable validators.

### Function: _invalid_if_no_key(func)

### Function: associate_with(message)

### Function: __init__(self, channel, seq, json)

### Function: __str__(self)

### Function: describe(self)

**Description:** A brief description of the message that is enough to identify it.

Examples:
'#1 request "launch" from IDE'
'#2 response to #1 request "launch" from IDE'.

### Function: payload(self)

**Description:** Payload of the message - self.body or self.arguments, depending on the
message type.

### Function: __call__(self)

**Description:** Same as self.payload(...).

### Function: __contains__(self, key)

**Description:** Same as (key in self.payload).

### Function: is_event(self)

**Description:** Returns True if this message is an Event of one of the specified types.

### Function: is_request(self)

**Description:** Returns True if this message is a Request of one of the specified types.

### Function: is_response(self)

**Description:** Returns True if this message is a Response to a request of one of the
specified types.

### Function: error(self, exc_type, format_string)

**Description:** Returns a new exception of the specified type from the point at which it is
invoked, with the specified formatted message as the reason.

The resulting exception will have its cause set to the Message object on which
error() was called. Additionally, if that message is a Request, a failure
response is immediately sent.

### Function: isnt_valid(self)

**Description:** Same as self.error(InvalidMessageError, ...).

### Function: cant_handle(self)

**Description:** Same as self.error(MessageHandlingError, ...).

### Function: __init__(self, channel, seq, event, body, json)

### Function: describe(self)

### Function: payload(self)

### Function: _parse(channel, message_dict)

### Function: _handle(self)

### Function: __init__(self, channel, seq, command, arguments, json)

### Function: describe(self)

### Function: payload(self)

### Function: respond(self, body)

### Function: _parse(channel, message_dict)

### Function: _handle(self)

### Function: __init__(self, channel, seq, command, arguments)

### Function: describe(self)

### Function: wait_for_response(self, raise_if_failed)

**Description:** Waits until a response is received for this request, records the Response
object for it in self.response, and returns response.body.

If no response was received from the other party before the channel closed,
self.response is a synthesized Response with body=NoMoreMessages().

If raise_if_failed=True and response.success is False, raises response.body
instead of returning.

### Function: on_response(self, response_handler)

**Description:** Registers a handler to invoke when a response is received for this request.
The handler is invoked with Response as its sole argument.

If response has already been received, invokes the handler immediately.

It is guaranteed that self.response is set before the handler is invoked.
If no response was received from the other party before the channel closed,
self.response is a dummy Response with body=NoMoreMessages().

The handler is always invoked asynchronously on an unspecified background
thread - thus, the caller of on_response() can never be blocked or deadlocked
by the handler.

No further incoming messages are processed until the handler returns, except for
responses to requests that have wait_for_response() invoked on them.

### Function: _enqueue_response_handlers(self)

### Function: __init__(self, channel, seq, request, body, json)

### Function: describe(self)

### Function: payload(self)

### Function: success(self)

**Description:** Whether the request succeeded or not.

### Function: result(self)

**Description:** Result of the request. Returns the value of response.body, unless it
is an exception, in which case it is raised instead.

### Function: _parse(channel, message_dict, body)

### Function: __init__(self, channel)

### Function: describe(self)

### Function: __init__(self, reason, cause, silent)

**Description:** Creates a new instance of this class, and immediately logs the exception.

Message handling errors are logged immediately unless silent=True, so that the
precise context in which they occured can be determined from the surrounding
log entries.

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __str__(self)

### Function: __repr__(self)

### Function: applies_to(self, message)

**Description:** Whether this MessageHandlingError can be treated as a reason why the
handling of message failed.

If self.cause is None, this is always true.

If self.cause is not None, this is only true if cause is message.

### Function: propagate(self, new_cause)

**Description:** Propagates this error, raising a new instance of the same class with the
same reason, but a different cause.

### Function: __str__(self)

### Function: __init__(self, stream, handlers, name)

### Function: __str__(self)

### Function: __repr__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, exc_tb)

### Function: close(self)

**Description:** Closes the underlying stream.

This does not immediately terminate any handlers that are already executing,
but they will be unable to respond. No new request or event handlers will
execute after this method is called, even for messages that have already been
received. However, response handlers will continue to executed for any request
that is still pending, as will any handlers registered via on_response().

### Function: start(self)

**Description:** Starts a message loop which parses incoming messages and invokes handlers
for them on a background thread, until the channel is closed.

Incoming messages, including responses to requests, will not be processed at
all until this is invoked.

### Function: wait(self)

**Description:** Waits for the message loop to terminate, and for all enqueued Response
message handlers to finish executing.

### Function: _prettify(self, message_dict)

**Description:** Reorders items in a MessageDict such that it is more readable.

### Function: _send_message(self, message)

**Description:** Sends a new message to the other party.

Generates a new sequence number for the message, and provides it to the
caller before the message is sent, using the context manager protocol::

    with send_message(...) as seq:
        # The message hasn't been sent yet.
        ...
    # Now the message has been sent.

Safe to call concurrently for the same channel from different threads.

### Function: send_request(self, command, arguments, on_before_send)

**Description:** Sends a new request, and returns the OutgoingRequest object for it.

If arguments is None or {}, "arguments" will be omitted in JSON.

If on_before_send is not None, invokes on_before_send() with the request
object as the sole argument, before the request actually gets sent.

Does not wait for response - use OutgoingRequest.wait_for_response().

Safe to call concurrently for the same channel from different threads.

### Function: send_event(self, event, body)

**Description:** Sends a new event.

If body is None or {}, "body" will be omitted in JSON.

Safe to call concurrently for the same channel from different threads.

### Function: request(self)

**Description:** Same as send_request(...).wait_for_response()

### Function: propagate(self, message)

**Description:** Sends a new message with the same type and payload.

If it was a request, returns the new OutgoingRequest object for it.

### Function: delegate(self, message)

**Description:** Like propagate(message).wait_for_response(), but will also propagate
any resulting MessageHandlingError back.

### Function: _parse_incoming_messages(self)

### Function: _parse_incoming_message(self)

**Description:** Reads incoming messages, parses them, and puts handlers into the queue
for _run_handlers() to invoke, until the channel is closed.

### Function: _enqueue_handlers(self, what)

**Description:** Enqueues handlers for _run_handlers() to run.

`what` is the Message being handled, and is used for logging purposes.

If the background thread with _run_handlers() isn't running yet, starts it.

### Function: _run_handlers(self)

**Description:** Runs enqueued handlers until the channel is closed, or until the handler
queue is empty once the channel is closed.

### Function: _get_handler_for(self, type, name)

**Description:** Returns the handler for a message of a given type.

### Function: _handle_disconnect(self)

### Function: __init__(self)

### Function: cleanup()

### Function: log_message_and_reraise_exception(format_string)

### Function: wrap(self, key)

### Function: run_handlers()

### Function: object_hook(d)

### Function: associate_with(message)
