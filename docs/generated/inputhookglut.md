## AI Summary

A file named inputhookglut.py.


### Function: glut_display()

### Function: glut_idle()

### Function: glut_close()

### Function: glut_int_handler(signum, frame)

### Function: inputhook_glut()

**Description:** Run the pyglet event loop by processing pending events only.

This keeps processing pending events until stdin is ready.  After
processing all pending events, a call to time.sleep is inserted.  This is
needed, otherwise, CPU usage is at 100%.  This sleep time should be tuned
though for best performance.
