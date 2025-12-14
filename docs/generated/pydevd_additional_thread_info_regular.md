## AI Summary

A file named pydevd_additional_thread_info_regular.py.


## Class: PyDBAdditionalThreadInfo

### Function: set_additional_thread_info(thread)

### Function: _update_stepping_info(info)

### Function: add_additional_info(info)

### Function: remove_additional_info(info)

### Function: any_thread_stepping()

### Function: __init__(self)

### Function: _get_related_thread(self)

### Function: _is_stepping(self)

### Function: get_topmost_frame(self, thread)

**Description:** Gets the topmost frame for the given thread. Note that it may be None
and callers should remove the reference to the frame as soon as possible
to avoid disturbing user code.

### Function: update_stepping_info(self)

### Function: __str__(self)
