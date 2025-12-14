## AI Summary

A file named debugger.py.


## Class: TerminalPdb

**Description:** Standalone IPython debugger.

### Function: set_trace(frame)

**Description:** Start debugging from `frame`.

If frame is not specified, debugging starts from caller's frame.

### Function: __init__(self)

### Function: pt_init(self, pt_session_options)

**Description:** Initialize the prompt session and the prompt loop
and store them in self.pt_app and self.pt_loop.

Additional keyword arguments for the PromptSession class
can be specified in pt_session_options.

### Function: _prompt(self)

**Description:** In case other prompt_toolkit apps have to run in parallel to this one (e.g. in madbg),
create_app_session must be used to prevent mixing up between them. According to the prompt_toolkit docs:

> If you need multiple applications running at the same time, you have to create a separate
> `AppSession` using a `with create_app_session():` block.

### Function: cmdloop(self, intro)

**Description:** Repeatedly issue a prompt, accept input, parse an initial prefix
off the received input, and dispatch to action methods, passing them
the remainder of the line as argument.

override the same methods from cmd.Cmd to provide prompt toolkit replacement.

### Function: do_interact(self, arg)

### Function: get_prompt_tokens()

### Function: gen_comp(self, text)
