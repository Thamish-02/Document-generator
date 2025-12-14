## AI Summary

A file named pydevd_console.py.


## Class: ConsoleMessage

**Description:** Console Messages

## Class: _DebugConsoleStdIn

## Class: DebugConsole

**Description:** Wrapper around code.InteractiveConsole, in order to send
errors and outputs to the debug console

## Class: InteractiveConsoleCache

### Function: get_interactive_console(thread_id, frame_id, frame, console_message)

**Description:** returns the global interactive console.
interactive console should have been initialized by this time
:rtype: DebugConsole

### Function: clear_interactive_console()

### Function: execute_console_command(frame, thread_id, frame_id, line, buffer_output)

**Description:** fetch an interactive console instance from the cache and
push the received command to the console.

create and return an instance of console_message

### Function: get_description(frame, thread_id, frame_id, expression)

### Function: get_completions(frame, act_tok)

**Description:** fetch all completions, create xml for the same
return the completions xml

### Function: __init__(self)

### Function: add_console_message(self, message_type, message)

**Description:** add messages in the console_messages list

### Function: update_more(self, more)

**Description:** more is set to true if further input is required from the user
else more is set to false

### Function: to_xml(self)

**Description:** Create an XML for console message_list, error and more (true/false)
<xml>
    <message_list>console message_list</message_list>
    <error>console error</error>
    <more>true/false</more>
</xml>

### Function: readline(self)

### Function: create_std_in(self)

### Function: push(self, line, frame, buffer_output)

**Description:** Change built-in stdout and stderr methods by the
new custom StdMessage.
execute the InteractiveConsole.push.
Change the stdout and stderr back be the original built-ins

:param buffer_output: if False won't redirect the output.

Return boolean (True if more input is required else False),
output_messages and input_messages

### Function: do_add_exec(self, line)

### Function: runcode(self, code)

**Description:** Execute a code object.

When an exception occurs, self.showtraceback() is called to
display a traceback.  All exceptions are caught except
SystemExit, which is reraised.

A note about KeyboardInterrupt: this exception may occur
elsewhere in this code, and may not always be caught.  The
caller should be prepared to deal with it.

### Function: get_namespace(self)
