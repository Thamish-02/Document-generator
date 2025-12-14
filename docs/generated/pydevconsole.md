## AI Summary

A file named pydevconsole.py.


## Class: Command

## Class: InterpreterInterface

**Description:** The methods in this class should be registered in the xml-rpc server.

## Class: _ProcessExecQueueHelper

### Function: set_debug_hook(debug_hook)

### Function: activate_mpl_if_already_imported(interpreter)

### Function: init_set_return_control_back(interpreter)

### Function: init_mpl_in_console(interpreter)

### Function: process_exec_queue(interpreter)

### Function: do_exit()

**Description:** We have to override the exit because calling sys.exit will only actually exit the main thread,
and as we're in a Xml-rpc server, that won't work.

### Function: start_console_server(host, port, interpreter)

### Function: start_server(host, port, client_port)

### Function: get_ipython_hidden_vars()

### Function: get_interpreter()

### Function: get_completions(text, token, globals, locals)

### Function: exec_code(code, globals, locals, debugger)

## Class: ConsoleWriter

### Function: console_exec(thread_id, frame_id, expression, dbg)

**Description:** returns 'False' in case expression is partially correct

### Function: __init__(self, interpreter, code_fragment)

**Description:** :type code_fragment: CodeFragment
:type interpreter: InteractiveConsole

### Function: symbol_for_fragment(code_fragment)

### Function: run(self)

### Function: __init__(self, host, client_port, mainThread, connect_status_queue)

### Function: do_add_exec(self, codeFragment)

### Function: get_namespace(self)

### Function: getCompletions(self, text, act_tok)

### Function: close(self)

### Function: get_greeting_msg(self)

### Function: return_control()

**Description:** A function that the inputhooks can call (via inputhook.stdin_ready()) to find
out if they should cede control and return

### Function: pid_exists(pid)

### Function: __init__(self, locals)

### Function: write(self, data)

### Function: showsyntaxerror(self, filename)

**Description:** Display the syntax error that just occurred.

### Function: showtraceback(self)

**Description:** Display the exception that just occurred.

### Function: pid_exists(pid)

### Function: pid_exists(pid)
