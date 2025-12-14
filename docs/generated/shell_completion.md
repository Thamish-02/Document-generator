## AI Summary

A file named shell_completion.py.


### Function: shell_complete(cli, ctx_args, prog_name, complete_var, instruction)

**Description:** Perform shell completion for the given CLI program.

:param cli: Command being called.
:param ctx_args: Extra arguments to pass to
    ``cli.make_context``.
:param prog_name: Name of the executable in the shell.
:param complete_var: Name of the environment variable that holds
    the completion instruction.
:param instruction: Value of ``complete_var`` with the completion
    instruction and shell, in the form ``instruction_shell``.
:return: Status code to exit with.

## Class: CompletionItem

**Description:** Represents a completion value and metadata about the value. The
default metadata is ``type`` to indicate special shell handling,
and ``help`` if a shell supports showing a help string next to the
value.

Arbitrary parameters can be passed when creating the object, and
accessed using ``item.attr``. If an attribute wasn't passed,
accessing it returns ``None``.

:param value: The completion suggestion.
:param type: Tells the shell script to provide special completion
    support for the type. Click uses ``"dir"`` and ``"file"``.
:param help: String shown next to the value if supported.
:param kwargs: Arbitrary metadata. The built-in implementations
    don't use this, but custom type completions paired with custom
    shell support could use it.

## Class: ShellComplete

**Description:** Base class for providing shell completion support. A subclass for
a given shell will override attributes and methods to implement the
completion instructions (``source`` and ``complete``).

:param cli: Command being called.
:param prog_name: Name of the executable in the shell.
:param complete_var: Name of the environment variable that holds
    the completion instruction.

.. versionadded:: 8.0

## Class: BashComplete

**Description:** Shell completion for Bash.

## Class: ZshComplete

**Description:** Shell completion for Zsh.

## Class: FishComplete

**Description:** Shell completion for Fish.

### Function: add_completion_class(cls, name)

**Description:** Register a :class:`ShellComplete` subclass under the given name.
The name will be provided by the completion instruction environment
variable during completion.

:param cls: The completion class that will handle completion for the
    shell.
:param name: Name to register the class under. Defaults to the
    class's ``name`` attribute.

### Function: get_completion_class(shell)

**Description:** Look up a registered :class:`ShellComplete` subclass by the name
provided by the completion instruction environment variable. If the
name isn't registered, returns ``None``.

:param shell: Name the class is registered under.

### Function: split_arg_string(string)

**Description:** Split an argument string as with :func:`shlex.split`, but don't
fail if the string is incomplete. Ignores a missing closing quote or
incomplete escape sequence and uses the partial token as-is.

.. code-block:: python

    split_arg_string("example 'my file")
    ["example", "my file"]

    split_arg_string("example my\")
    ["example", "my"]

:param string: String to split.

.. versionchanged:: 8.2
    Moved to ``shell_completion`` from ``parser``.

### Function: _is_incomplete_argument(ctx, param)

**Description:** Determine if the given parameter is an argument that can still
accept values.

:param ctx: Invocation context for the command represented by the
    parsed complete args.
:param param: Argument object being checked.

### Function: _start_of_option(ctx, value)

**Description:** Check if the value looks like the start of an option.

### Function: _is_incomplete_option(ctx, args, param)

**Description:** Determine if the given parameter is an option that needs a value.

:param args: List of complete args before the incomplete value.
:param param: Option object being checked.

### Function: _resolve_context(cli, ctx_args, prog_name, args)

**Description:** Produce the context hierarchy starting with the command and
traversing the complete arguments. This only follows the commands,
it doesn't trigger input prompts or callbacks.

:param cli: Command being called.
:param prog_name: Name of the executable in the shell.
:param args: List of complete args before the incomplete value.

### Function: _resolve_incomplete(ctx, args, incomplete)

**Description:** Find the Click object that will handle the completion of the
incomplete value. Return the object and the incomplete value.

:param ctx: Invocation context for the command represented by
    the parsed complete args.
:param args: List of complete args before the incomplete value.
:param incomplete: Value being completed. May be empty.

### Function: __init__(self, value, type, help)

### Function: __getattr__(self, name)

### Function: __init__(self, cli, ctx_args, prog_name, complete_var)

### Function: func_name(self)

**Description:** The name of the shell function defined by the completion
script.

### Function: source_vars(self)

**Description:** Vars for formatting :attr:`source_template`.

By default this provides ``complete_func``, ``complete_var``,
and ``prog_name``.

### Function: source(self)

**Description:** Produce the shell script that defines the completion
function. By default this ``%``-style formats
:attr:`source_template` with the dict returned by
:meth:`source_vars`.

### Function: get_completion_args(self)

**Description:** Use the env vars defined by the shell script to return a
tuple of ``args, incomplete``. This must be implemented by
subclasses.

### Function: get_completions(self, args, incomplete)

**Description:** Determine the context and last complete command or parameter
from the complete args. Call that object's ``shell_complete``
method to get the completions for the incomplete value.

:param args: List of complete args before the incomplete value.
:param incomplete: Value being completed. May be empty.

### Function: format_completion(self, item)

**Description:** Format a completion item into the form recognized by the
shell script. This must be implemented by subclasses.

:param item: Completion item to format.

### Function: complete(self)

**Description:** Produce the completion data to send back to the shell.

By default this calls :meth:`get_completion_args`, gets the
completions, then calls :meth:`format_completion` for each
completion.

### Function: _check_version()

### Function: source(self)

### Function: get_completion_args(self)

### Function: format_completion(self, item)

### Function: get_completion_args(self)

### Function: format_completion(self, item)

### Function: get_completion_args(self)

### Function: format_completion(self, item)
