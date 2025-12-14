## AI Summary

A file named inputhookpyglet.py.


### Function: inputhook_pyglet()

**Description:** Run the pyglet event loop by processing pending events only.

This keeps processing pending events until stdin is ready.  After
processing all pending events, a call to time.sleep is inserted.  This is
needed, otherwise, CPU usage is at 100%.  This sleep time should be tuned
though for best performance.

### Function: flip(window)

### Function: flip(window)
