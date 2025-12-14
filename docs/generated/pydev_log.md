## AI Summary

A file named pydev_log.py.


## Class: _LoggingGlobals

### Function: initialize_debug_stream(reinitialize)

**Description:** :param bool reinitialize:
    Reinitialize is used to update the debug stream after a fork (thus, if it wasn't
    initialized, we don't need to do anything, just wait for the first regular log call
    to initialize).

### Function: _compute_filename_with_pid(target_file, pid)

### Function: log_to(log_file, log_level)

### Function: list_log_files(pydevd_debug_file)

### Function: log_context(trace_level, stream)

**Description:** To be used to temporarily change the logging settings.

### Function: _pydevd_log(level, msg)

**Description:** Levels are:

0 most serious warnings/errors (always printed)
1 warnings/significant events
2 informational trace
3 verbose mode

### Function: _pydevd_log_exception(msg)

### Function: verbose(msg)

### Function: debug(msg)

### Function: info(msg)

### Function: critical(msg)

### Function: exception(msg)

### Function: error_once(msg)

### Function: exception_once(msg)

### Function: debug_once(msg)

### Function: show_compile_cython_command_line()
