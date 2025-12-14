## AI Summary

A file named nest_asyncio.py.


### Function: apply(loop)

**Description:** Patch asyncio to make its event loop reentrant.

### Function: _patch_asyncio()

**Description:** Patch asyncio module to use pure Python tasks and futures.

### Function: _patch_policy()

**Description:** Patch the policy to always return a patched loop.

### Function: _patch_loop(loop)

**Description:** Patch loop to make it reentrant.

### Function: _patch_tornado()

**Description:** If tornado is imported before nest_asyncio, make tornado aware of
the pure-Python asyncio Future.

### Function: run(main)

### Function: _get_event_loop(stacklevel)

### Function: get_event_loop(self)

### Function: run_forever(self)

### Function: run_until_complete(self, future)

### Function: _run_once(self)

**Description:** Simplified re-implementation of asyncio's _run_once that
runs handles as they become ready.

### Function: manage_run(self)

**Description:** Set up the loop for running.

### Function: manage_asyncgens(self)

### Function: _check_running(self)

**Description:** Do not throw exception if loop is already running.
