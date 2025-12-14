## AI Summary

A file named pydevd_utils.py.


### Function: save_main_module(file, module_name)

### Function: is_current_thread_main_thread()

### Function: get_main_thread()

### Function: to_number(x)

### Function: compare_object_attrs_key(x)

### Function: is_string(x)

### Function: to_string(x)

### Function: print_exc()

### Function: quote_smart(s, safe)

### Function: get_clsname_for_code(code, frame)

### Function: get_non_pydevd_threads()

### Function: dump_threads(stream, show_pydevd_threads)

**Description:** Helper to dump thread info.

### Function: _extract_variable_nested_braces(char_iter)

### Function: _extract_expression_list(log_message)

### Function: convert_dap_log_message_to_expression(log_message)

### Function: notify_about_gevent_if_needed(stream)

**Description:** When debugging with gevent check that the gevent flag is used if the user uses the gevent
monkey-patching.

:return bool:
    Returns True if a message had to be shown to the user and False otherwise.

### Function: hasattr_checked(obj, name)

### Function: getattr_checked(obj, name)

### Function: dir_checked(obj)

### Function: isinstance_checked(obj, cls)

## Class: ScopeRequest

## Class: DAPGrouper

**Description:** Note: this is a helper class to group variables on the debug adapter protocol (DAP). For
the xml protocol the type is just added to each variable and the UI can group/hide it as needed.

### Function: interrupt_main_thread(main_thread)

**Description:** Generates a KeyboardInterrupt in the main thread by sending a Ctrl+C
or by calling thread.interrupt_main().

:param main_thread:
    Needed because Jython needs main_thread._thread.interrupt() to be called.

Note: if unable to send a Ctrl+C, the KeyboardInterrupt will only be raised
when the next Python instruction is about to be executed (so, it won't interrupt
a sleep(1000)).

## Class: Timer

### Function: import_attr_from_module(import_with_attr_access)

### Function: __init__(self, variable_reference, scope)

### Function: __eq__(self, o)

### Function: __ne__(self, o)

### Function: __hash__(self)

### Function: __init__(self, scope)

### Function: get_contents_debug_adapter_protocol(self)

### Function: __eq__(self, o)

### Function: __ne__(self, o)

### Function: __hash__(self)

### Function: __repr__(self)

### Function: __str__(self)

### Function: __init__(self, min_diff)

### Function: print_time(self, msg)

### Function: _report_slow(self, compute_msg)

### Function: report_if_compute_repr_attr_slow(self, attrs_tab_separated, attr_name, attr_type)

### Function: _compute_repr_slow(self, diff, attrs_tab_separated, attr_name, attr_type)

### Function: report_if_getting_attr_slow(self, cls, attr_name)

### Function: _compute_get_attr_slow(self, diff, cls, attr_name)
