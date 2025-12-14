## AI Summary

A file named add_code_to_python_process.py.


### Function: _create_win_event(name)

### Function: is_python_64bit()

### Function: get_target_filename(is_target_process_64, prefix, extension)

### Function: run_python_code_windows(pid, python_code, connect_debugger_tracing, show_debug_info)

### Function: _acquire_mutex(mutex_name, timeout)

**Description:** Only one process may be attaching to a pid, so, create a system mutex
to make sure this holds in practice.

### Function: _win_write_to_shared_named_memory(python_code, pid)

### Function: run_python_code_linux(pid, python_code, connect_debugger_tracing, show_debug_info)

### Function: find_helper_script(filedir, script_name)

### Function: run_python_code_mac(pid, python_code, connect_debugger_tracing, show_debug_info)

### Function: test()

### Function: main(args)

## Class: _WinEvent

## Class: TimeoutError

### Function: wait_for_event_set(self, timeout)

**Description:** :param timeout: in seconds

### Function: run_python_code()
