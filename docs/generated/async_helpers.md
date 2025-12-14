## AI Summary

A file named async_helpers.py.


### Function: get_asyncio_loop()

**Description:** asyncio has deprecated get_event_loop

Replicate it here, with our desired semantics:

- always returns a valid, not-closed loop
- not thread-local like asyncio's,
  because we only want one loop for IPython
- if called from inside a coroutine (e.g. in ipykernel),
  return the running loop

.. versionadded:: 8.0

## Class: _AsyncIORunner

## Class: _AsyncIOProxy

**Description:** Proxy-object for an asyncio

Any coroutine methods will be wrapped in event_loop.run_

### Function: _curio_runner(coroutine)

**Description:** handler for curio autoawait

### Function: _trio_runner(async_fn)

### Function: _pseudo_sync_runner(coro)

**Description:** A runner that does not really allow async execution, and just advance the coroutine.

See discussion in https://github.com/python-trio/trio/issues/608,

Credit to Nathaniel Smith

### Function: _should_be_async(cell)

**Description:** Detect if a block of code need to be wrapped in an `async def`

Attempt to parse the block of code, it it compile we're fine.
Otherwise we  wrap if and try to compile.

If it works, assume it should be async. Otherwise Return False.

Not handled yet: If the block of code has a return statement as the top
level, it will be seen as async. This is a know limitation.

### Function: __call__(self, coro)

**Description:** Handler for asyncio autoawait

### Function: __str__(self)

### Function: __init__(self, obj, event_loop)

### Function: __repr__(self)

### Function: __getattr__(self, key)

### Function: __dir__(self)

### Function: _wrapped()
