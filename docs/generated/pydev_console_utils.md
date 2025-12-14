## AI Summary

A file named pydev_console_utils.py.


## Class: BaseStdIn

## Class: StdIn

**Description:** Object to be added to stdin (to emulate it as non-blocking while the next line arrives)

## Class: DebugConsoleStdIn

**Description:** Object to be added to stdin (to emulate it as non-blocking while the next line arrives)

## Class: CodeFragment

## Class: BaseInterpreterInterface

## Class: FakeFrame

**Description:** Used to show console with variables connection.
A class to be used as a mock of a frame.

### Function: __init__(self, original_stdin)

### Function: readline(self)

### Function: write(self)

### Function: flush(self)

### Function: read(self)

### Function: close(self)

### Function: __iter__(self)

### Function: __getattr__(self, item)

### Function: __init__(self, interpreter, host, client_port, original_stdin)

### Function: readline(self)

### Function: close(self)

### Function: __init__(self, py_db, original_stdin)

**Description:** :param py_db:
    If None, get_global_debugger() is used.

### Function: __send_input_requested_message(self, is_started)

### Function: notify_input_requested(self)

### Function: readline(self)

### Function: read(self)

### Function: __init__(self, text, is_single_line)

### Function: append(self, code_fragment)

### Function: __init__(self, mainThread, connect_status_queue)

### Function: build_banner(self)

### Function: get_greeting_msg(self)

### Function: init_mpl_modules_for_patching(self)

### Function: need_more_for_code(self, source)

### Function: need_more(self, code_fragment)

### Function: create_std_in(self, debugger, original_std_in)

### Function: add_exec(self, code_fragment, debugger)

### Function: do_add_exec(self, codeFragment)

**Description:** Subclasses should override.

@return: more (True if more input is needed to complete the statement and False if the statement is complete).

### Function: get_namespace(self)

**Description:** Subclasses should override.

@return: dict with namespace.

### Function: __resolve_reference__(self, text)

**Description:** :type text: str

### Function: getDescription(self, text)

### Function: do_exec_code(self, code, is_single_line)

### Function: execLine(self, line)

### Function: execMultipleLines(self, lines)

### Function: interrupt(self)

### Function: close(self)

### Function: start_exec(self)

### Function: get_server(self)

### Function: ShowConsole(self)

### Function: finish_exec(self, more)

### Function: getFrame(self)

### Function: getVariable(self, attributes)

### Function: getArray(self, attr, roffset, coffset, rows, cols, format)

### Function: evaluate(self, expression)

### Function: loadFullValue(self, seq, scope_attrs)

**Description:** Evaluate full value for async Console variables in a separate thread and send results to IDE side
:param seq: id of command
:param scope_attrs: a sequence of variables with their attributes separated by NEXT_VALUE_SEPARATOR
(i.e.: obj      attr1   attr2NEXT_VALUE_SEPARATORobj2ttr1      attr2)
:return:

### Function: changeVariable(self, attr, value)

### Function: connectToDebugger(self, debuggerPort, debugger_options)

**Description:** Used to show console with variables connection.
Mainly, monkey-patches things in the debugger structure so that the debugger protocol works.

### Function: handshake(self)

### Function: get_connect_status_queue(self)

### Function: hello(self, input_str)

### Function: enableGui(self, guiname)

**Description:** Enable the GUI specified in guiname (see inputhook for list).
As with IPython, enabling multiple GUIs isn't an error, but
only the last one's main loop runs and it may not work

### Function: get_ipython_hidden_vars_dict(self)

### Function: do_change_variable()

### Function: do_connect_to_debugger()

### Function: do_enable_gui()
