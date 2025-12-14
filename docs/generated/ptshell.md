## AI Summary

A file named ptshell.py.


### Function: ask_yes_no(prompt, default, interrupt)

**Description:** Asks a question and returns a boolean (y/n) answer.

If default is given (one of 'y','n'), it is used if the user input is
empty. If interrupt is given (one of 'y','n'), it is used if the user
presses Ctrl-C. Otherwise the question is repeated until an answer is
given.

An EOF is treated as the default answer.  If there is no default, an
exception is raised to prevent infinite loops.

Valid answers are: y/yes/n/no (match is not case sensitive).

### Function: get_pygments_lexer(name)

## Class: JupyterPTCompleter

**Description:** Adaptor to provide kernel completions to prompt_toolkit

## Class: ZMQTerminalInteractiveShell

### Function: __init__(self, jup_completer)

### Function: get_completions(self, document, complete_event)

### Function: _client_changed(self, name, old, new)

### Function: _banner1_default(self)

### Function: __init__(self)

### Function: init_completer(self)

**Description:** Initialize the completion machinery.

This creates completion machinery that can be used by client code,
either interactively in-process (typically triggered by the readline
library), programmatically (such as in test suites) or out-of-process
(typically over the network by remote frontends).

### Function: init_history(self)

**Description:** Sets up the command history. 

### Function: vi_mode(self)

### Function: get_prompt_tokens(self, ec)

### Function: get_continuation_tokens(self, width)

### Function: get_out_prompt_tokens(self)

### Function: print_out_prompt(self)

### Function: get_remote_prompt_tokens(self)

### Function: print_remote_prompt(self, ec)

### Function: pt_complete_style(self)

### Function: init_kernel_info(self)

**Description:** Wait for a kernel to be ready, and store kernel info

### Function: show_banner(self)

### Function: init_prompt_toolkit_cli(self)

### Function: init_io(self)

### Function: check_complete(self, code)

### Function: ask_exit(self)

### Function: pre_prompt(self)

### Function: mainloop(self)

### Function: run_cell(self, cell, store_history)

**Description:** Run a complete IPython cell.

Parameters
----------
cell : str
  The code (including IPython code such as %magic functions) to run.
store_history : bool
  If True, the raw and translated cell will be stored in IPython's
  history. For user code calling back into IPython's machinery, this
  should be set to False.

### Function: handle_execute_reply(self, msg_id, timeout)

### Function: handle_is_complete_reply(self, msg_id, timeout)

**Description:** Wait for a repsonse from the kernel, and return two values:
    more? - (boolean) should the frontend ask for more input
    indent - an indent string to prefix the input
Overloaded methods may want to examine the comeplete source. Its is
in the self._source_lines_buffered list.

### Function: from_here(self, msg)

**Description:** Return whether a message is from this session

### Function: include_output(self, msg)

**Description:** Return whether we should include a given output message

### Function: handle_iopub(self, msg_id)

**Description:** Process messages on the IOPub channel

This method consumes and processes messages on the IOPub channel,
such as stdout, stderr, execute_result and status.

It only displays output that is caused by this session.

### Function: handle_rich_data(self, data)

### Function: handle_image(self, data, mime)

### Function: handle_image_PIL(self, data, mime)

### Function: handle_image_stream(self, data, mime)

### Function: handle_image_tempfile(self, data, mime)

### Function: handle_image_callable(self, data, mime)

### Function: handle_input_request(self, msg_id, timeout)

**Description:** Method to capture raw_input
        

### Function: _(event)

### Function: _(event)

### Function: _(event)

### Function: _(event)

### Function: _(event)

### Function: set_doc()

### Function: double_int(sig, frame)
