## AI Summary

A file named pydevd_custom_frames.py.


## Class: CustomFramesContainer

### Function: custom_frames_container_init()

## Class: CustomFrame

### Function: add_custom_frame(frame, name, thread_id)

**Description:** It's possible to show paused frames by adding a custom frame through this API (it's
intended to be used for coroutines, but could potentially be used for generators too).

:param frame:
    The topmost frame to be shown paused when a thread with thread.ident == thread_id is paused.

:param name:
    The name to be shown for the custom thread in the UI.

:param thread_id:
    The thread id to which this frame is related (must match thread.ident).

:return: str
    Returns the custom thread id which will be used to show the given frame paused.

### Function: update_custom_frame(frame_custom_thread_id, frame, thread_id, name)

### Function: remove_custom_frame(frame_custom_thread_id)

### Function: __init__(self, name, frame, thread_id)
