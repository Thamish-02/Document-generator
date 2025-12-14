## AI Summary

A file named pydevd_save_locals.py.


### Function: is_save_locals_available()

### Function: save_locals(frame)

**Description:** Copy values from locals_dict into the fast stack slots in the given frame.

Note: the 'save_locals' branch had a different approach wrapping the frame (much more code, but it gives ideas
on how to save things partially, not the 'whole' locals).

### Function: make_save_locals_impl()

**Description:** Factory for the 'save_locals_impl' method. This may seem like a complicated pattern but it is essential that the method is created at
module load time. Inner imports after module load time would cause an occasional debugger deadlock due to the importer lock and debugger
lock being taken in different order in  different threads.

### Function: update_globals_and_locals(updated_globals, initial_globals, frame)

### Function: save_locals_ctypes_impl(frame)

### Function: save_locals_pypy_impl(frame)
