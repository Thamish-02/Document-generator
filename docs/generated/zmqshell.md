## AI Summary

A file named zmqshell.py.


## Class: ZMQDisplayPublisher

**Description:** A display publisher that publishes data using a ZeroMQ PUB socket.

## Class: KernelMagics

**Description:** Kernel magics.

## Class: ZMQInteractiveShell

**Description:** A subclass of InteractiveShell for ZMQ.

### Function: __init__(self)

### Function: parent_header(self)

### Function: set_parent(self, parent)

**Description:** Set the parent for outbound messages.

### Function: _flush_streams(self)

**Description:** flush IO Streams prior to display

### Function: _default_thread_local(self)

**Description:** Initialize our thread local storage

### Function: _hooks(self)

### Function: publish(self, data, metadata)

**Description:** Publish a display-data message

Parameters
----------
data : dict
    A mime-bundle dict, keyed by mime-type.
metadata : dict, optional
    Metadata associated with the data.
transient : dict, optional, keyword-only
    Transient data that may only be relevant during a live display,
    such as display_id.
    Transient data should not be persisted to documents.
update : bool, optional, keyword-only
    If True, send an update_display_data message instead of display_data.

### Function: clear_output(self, wait)

**Description:** Clear output associated with the current execution (cell).

Parameters
----------
wait : bool (default: False)
    If True, the output will not be cleared immediately,
    instead waiting for the next display before clearing.
    This reduces bounce during repeated clear & display loops.

### Function: register_hook(self, hook)

**Description:** Registers a hook with the thread-local storage.

Parameters
----------
hook : Any callable object

Returns
-------
Either a publishable message, or `None`.
The DisplayHook objects must return a message from
the __call__ method if they still require the
`session.send` method to be called after transformation.
Returning `None` will halt that execution path, and
session.send will not be called.

### Function: unregister_hook(self, hook)

**Description:** Un-registers a hook with the thread-local storage.

Parameters
----------
hook : Any callable object which has previously been
    registered as a hook.

Returns
-------
bool - `True` if the hook was removed, `False` if it wasn't
    found.

### Function: edit(self, parameter_s, last_call)

**Description:** Bring up an editor and execute the resulting code.

Usage:
  %edit [options] [args]

%edit runs an external text editor. You will need to set the command for
this editor via the ``TerminalInteractiveShell.editor`` option in your
configuration file before it will work.

This command allows you to conveniently edit multi-line code right in
your IPython session.

If called without arguments, %edit opens up an empty editor with a
temporary file and will execute the contents of this file when you
close it (don't forget to save it!).

Options:

-n <number>
  Open the editor at a specified line number. By default, the IPython
  editor hook uses the unix syntax 'editor +N filename', but you can
  configure this by providing your own modified hook if your favorite
  editor supports line-number specifications with a different syntax.

-p
  Call the editor with the same data as the previous time it was used,
  regardless of how long ago (in your current session) it was.

-r
  Use 'raw' input. This option only applies to input taken from the
  user's history.  By default, the 'processed' history is used, so that
  magics are loaded in their transformed version to valid Python.  If
  this option is given, the raw input as typed as the command line is
  used instead.  When you exit the editor, it will be executed by
  IPython's own processor.

Arguments:

If arguments are given, the following possibilities exist:

- The arguments are numbers or pairs of colon-separated numbers (like
  1 4:8 9). These are interpreted as lines of previous input to be
  loaded into the editor. The syntax is the same of the %macro command.

- If the argument doesn't start with a number, it is evaluated as a
  variable and its contents loaded into the editor. You can thus edit
  any string which contains python code (including the result of
  previous edits).

- If the argument is the name of an object (other than a string),
  IPython will try to locate the file where it was defined and open the
  editor at the point where it is defined. You can use ``%edit function``
  to load an editor exactly at the point where 'function' is defined,
  edit it and have the file be executed automatically.

  If the object is a macro (see %macro for details), this opens up your
  specified editor with a temporary file containing the macro's data.
  Upon exit, the macro is reloaded with the contents of the file.

  Note: opening at an exact line is only supported under Unix, and some
  editors (like kedit and gedit up to Gnome 2.8) do not understand the
  '+NUMBER' parameter necessary for this feature. Good editors like
  (X)Emacs, vi, jed, pico and joe all do.

- If the argument is not found as a variable, IPython will look for a
  file with that name (adding .py if necessary) and load it into the
  editor. It will execute its contents with execfile() when you exit,
  loading any code in the file into your interactive namespace.

Unlike in the terminal, this is designed to use a GUI editor, and we do
not know when it has closed. So the file you edit will not be
automatically executed or printed.

Note that %edit is also available through the alias %ed.

### Function: clear(self, arg_s)

**Description:** Clear the terminal.

### Function: less(self, arg_s)

**Description:** Show a file through the pager.

Files ending in .py are syntax-highlighted.

### Function: connect_info(self, arg_s)

**Description:** Print information for connecting other clients to this kernel

It will print the contents of this session's connection file, as well as
shortcuts for local clients.

In the simplest case, when called from the most recently launched kernel,
secondary clients can be connected, simply with:

$> jupyter <app> --existing

### Function: qtconsole(self, arg_s)

**Description:** Open a qtconsole connected to this kernel.

Useful for connecting a qtconsole to running notebooks, for better
debugging.

### Function: autosave(self, arg_s)

**Description:** Set the autosave interval in the notebook (in seconds).

The default value is 120, or two minutes.
``%autosave 0`` will disable autosave.

This magic only has an effect when called from the notebook interface.
It has no effect when called in a startup file.

### Function: subshell(self, arg_s)

**Description:** List all current subshells

### Function: __init__(self)

### Function: _default_banner1(self)

### Function: _default_exiter(self)

### Function: _update_exit_now(self, change)

**Description:** stop eventloop when exit_now fires

### Function: enable_gui(self, gui)

**Description:** Enable a given gui.

### Function: init_environment(self)

**Description:** Configure the user's environment.

### Function: payloadpage_page(self, strg, start, screen_lines, pager_cmd)

**Description:** Print a string, piping through a pager.

This version ignores the screen_lines and pager_cmd arguments and uses
IPython's payload system instead.

Parameters
----------
strg : str or mime-dict
    Text to page, or a mime-type keyed dict of already formatted data.
start : int
    Starting line at which to place the display.

### Function: init_hooks(self)

**Description:** Initialize hooks.

### Function: init_data_pub(self)

**Description:** Delay datapub init until request, for deprecation warnings

### Function: data_pub(self)

### Function: data_pub(self, pub)

### Function: ask_exit(self)

**Description:** Engage the exit actions.

### Function: run_cell(self)

**Description:** Run a cell.

### Function: _showtraceback(self, etype, evalue, stb)

### Function: set_next_input(self, text, replace)

**Description:** Send the specified text to the frontend to be presented at the next
input cell.

### Function: parent_header(self)

### Function: parent_header(self, value)

### Function: set_parent(self, parent)

**Description:** Set the parent header for associating output with its triggering input

When called from a thread, sets the thread-local value, which persists
until the next call from this thread.

### Function: get_parent(self)

**Description:** Get the parent header.

If set_parent has never been called from the current thread,
the value from the last call to set_parent from _any_ thread will be used
(typically the currently running cell).

### Function: init_magics(self)

**Description:** Initialize magics.

### Function: init_virtualenv(self)

**Description:** Initialize virtual environment.

### Function: system_piped(self, cmd)

**Description:** Call the given cmd in a subprocess, piping stdout/err

Parameters
----------
cmd : str
    Command to execute (can not end in '&', as background processes are
    not supported.  Should not be a command that expects input
    other than simple text.

### Function: man(self, arg_s)

**Description:** Find the man page for the given command and display in pager.
