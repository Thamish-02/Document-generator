## AI Summary

A file named _completions.py.


## Class: ChatCompletionStream

**Description:** Wrapper over the Chat Completions streaming API that adds helpful
events such as `content.done`, supports automatically parsing
responses & tool calls and accumulates a `ChatCompletion` object
from each individual chunk.

https://platform.openai.com/docs/api-reference/streaming

## Class: ChatCompletionStreamManager

**Description:** Context manager over a `ChatCompletionStream` that is returned by `.stream()`.

This context manager ensures the response cannot be leaked if you don't read
the stream to completion.

Usage:
```py
with client.chat.completions.stream(...) as stream:
    for event in stream:
        ...
```

## Class: AsyncChatCompletionStream

**Description:** Wrapper over the Chat Completions streaming API that adds helpful
events such as `content.done`, supports automatically parsing
responses & tool calls and accumulates a `ChatCompletion` object
from each individual chunk.

https://platform.openai.com/docs/api-reference/streaming

## Class: AsyncChatCompletionStreamManager

**Description:** Context manager over a `AsyncChatCompletionStream` that is returned by `.stream()`.

This context manager ensures the response cannot be leaked if you don't read
the stream to completion.

Usage:
```py
async with client.chat.completions.stream(...) as stream:
    for event in stream:
        ...
```

## Class: ChatCompletionStreamState

**Description:** Helper class for manually accumulating `ChatCompletionChunk`s into a final `ChatCompletion` object.

This is useful in cases where you can't always use the `.stream()` method, e.g.

```py
from openai.lib.streaming.chat import ChatCompletionStreamState

state = ChatCompletionStreamState()

stream = client.chat.completions.create(..., stream=True)
for chunk in response:
    state.handle_chunk(chunk)

    # can also access the accumulated `ChatCompletion` mid-stream
    state.current_completion_snapshot

print(state.get_final_completion())
```

## Class: ChoiceEventState

### Function: _convert_initial_chunk_into_snapshot(chunk)

### Function: _is_valid_chat_completion_chunk_weak(sse_event)

### Function: __init__(self)

### Function: __next__(self)

### Function: __iter__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc, exc_tb)

### Function: close(self)

**Description:** Close the response and release the connection.

Automatically called if the response body is read to completion.

### Function: get_final_completion(self)

**Description:** Waits until the stream has been read to completion and returns
the accumulated `ParsedChatCompletion` object.

If you passed a class type to `.stream()`, the `completion.choices[0].message.parsed`
property will be the content deserialised into that class, if there was any content returned
by the API.

### Function: until_done(self)

**Description:** Blocks until the stream has been consumed.

### Function: current_completion_snapshot(self)

### Function: __stream__(self)

### Function: __init__(self, api_request)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc, exc_tb)

### Function: __init__(self)

### Function: current_completion_snapshot(self)

### Function: __init__(self, api_request)

### Function: __init__(self)

### Function: get_final_completion(self)

**Description:** Parse the final completion object.

Note this does not provide any guarantees that the stream has actually finished, you must
only call this method when the stream is finished.

### Function: current_completion_snapshot(self)

### Function: handle_chunk(self, chunk)

**Description:** Accumulate a new chunk into the snapshot and returns an iterable of events to yield.

### Function: _get_choice_state(self, choice)

### Function: _accumulate_chunk(self, chunk)

### Function: _build_events(self)

### Function: __init__(self)

### Function: get_done_events(self)

### Function: _content_done_events(self)

### Function: _add_tool_done_event(self)
