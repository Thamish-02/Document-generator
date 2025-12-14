## AI Summary

A file named _assistants.py.


## Class: AssistantEventHandler

## Class: AssistantStreamManager

**Description:** Wrapper over AssistantStreamEventHandler that is returned by `.stream()`
so that a context manager can be used.

```py
with client.threads.create_and_run_stream(...) as stream:
    for event in stream:
        ...
```

## Class: AsyncAssistantEventHandler

## Class: AsyncAssistantStreamManager

**Description:** Wrapper over AsyncAssistantStreamEventHandler that is returned by `.stream()`
so that an async context manager can be used without `await`ing the
original client call.

```py
async with client.threads.create_and_run_stream(...) as stream:
    async for event in stream:
        ...
```

### Function: accumulate_run_step()

### Function: accumulate_event()

**Description:** Returns a tuple of message snapshot and newly created text message deltas

### Function: accumulate_delta(acc, delta)

### Function: __init__(self)

### Function: _init(self, stream)

### Function: __next__(self)

### Function: __iter__(self)

### Function: current_event(self)

### Function: current_run(self)

### Function: current_run_step_snapshot(self)

### Function: current_message_snapshot(self)

### Function: close(self)

**Description:** Close the response and release the connection.

Automatically called when the context manager exits.

### Function: until_done(self)

**Description:** Waits until the stream has been consumed

### Function: get_final_run(self)

**Description:** Wait for the stream to finish and returns the completed Run object

### Function: get_final_run_steps(self)

**Description:** Wait for the stream to finish and returns the steps taken in this run

### Function: get_final_messages(self)

**Description:** Wait for the stream to finish and returns the messages emitted in this run

### Function: __text_deltas__(self)

### Function: on_end(self)

**Description:** Fires when the stream has finished.

This happens if the stream is read to completion
or if an exception occurs during iteration.

### Function: on_event(self, event)

**Description:** Callback that is fired for every Server-Sent-Event

### Function: on_run_step_created(self, run_step)

**Description:** Callback that is fired when a run step is created

### Function: on_run_step_delta(self, delta, snapshot)

**Description:** Callback that is fired whenever a run step delta is returned from the API

The first argument is just the delta as sent by the API and the second argument
is the accumulated snapshot of the run step. For example, a tool calls event may
look like this:

# delta
tool_calls=[
    RunStepDeltaToolCallsCodeInterpreter(
        index=0,
        type='code_interpreter',
        id=None,
        code_interpreter=CodeInterpreter(input=' sympy', outputs=None)
    )
]
# snapshot
tool_calls=[
    CodeToolCall(
        id='call_wKayJlcYV12NiadiZuJXxcfx',
        code_interpreter=CodeInterpreter(input='from sympy', outputs=[]),
        type='code_interpreter',
        index=0
    )
],

### Function: on_run_step_done(self, run_step)

**Description:** Callback that is fired when a run step is completed

### Function: on_tool_call_created(self, tool_call)

**Description:** Callback that is fired when a tool call is created

### Function: on_tool_call_delta(self, delta, snapshot)

**Description:** Callback that is fired when a tool call delta is encountered

### Function: on_tool_call_done(self, tool_call)

**Description:** Callback that is fired when a tool call delta is encountered

### Function: on_exception(self, exception)

**Description:** Fired whenever an exception happens during streaming

### Function: on_timeout(self)

**Description:** Fires if the request times out

### Function: on_message_created(self, message)

**Description:** Callback that is fired when a message is created

### Function: on_message_delta(self, delta, snapshot)

**Description:** Callback that is fired whenever a message delta is returned from the API

The first argument is just the delta as sent by the API and the second argument
is the accumulated snapshot of the message. For example, a text content event may
look like this:

# delta
MessageDeltaText(
    index=0,
    type='text',
    text=Text(
        value=' Jane'
    ),
)
# snapshot
MessageContentText(
    index=0,
    type='text',
    text=Text(
        value='Certainly, Jane'
    ),
)

### Function: on_message_done(self, message)

**Description:** Callback that is fired when a message is completed

### Function: on_text_created(self, text)

**Description:** Callback that is fired when a text content block is created

### Function: on_text_delta(self, delta, snapshot)

**Description:** Callback that is fired whenever a text content delta is returned
by the API.

The first argument is just the delta as sent by the API and the second argument
is the accumulated snapshot of the text. For example:

on_text_delta(TextDelta(value="The"), Text(value="The")),
on_text_delta(TextDelta(value=" solution"), Text(value="The solution")),
on_text_delta(TextDelta(value=" to"), Text(value="The solution to")),
on_text_delta(TextDelta(value=" the"), Text(value="The solution to the")),
on_text_delta(TextDelta(value=" equation"), Text(value="The solution to the equation")),

### Function: on_text_done(self, text)

**Description:** Callback that is fired when a text content block is finished

### Function: on_image_file_done(self, image_file)

**Description:** Callback that is fired when an image file block is finished

### Function: _emit_sse_event(self, event)

### Function: __stream__(self)

### Function: __init__(self, api_request)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc, exc_tb)

### Function: __init__(self)

### Function: _init(self, stream)

### Function: current_event(self)

### Function: current_run(self)

### Function: current_run_step_snapshot(self)

### Function: current_message_snapshot(self)

### Function: __init__(self, api_request)
