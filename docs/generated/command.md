## AI Summary

A file named command.py.


## Class: JupyterParser

**Description:** A Jupyter argument parser.

### Function: jupyter_parser()

**Description:** Create a jupyter parser object.

### Function: list_subcommands()

**Description:** List all jupyter subcommands

searches PATH for `jupyter-name`

Returns a list of jupyter's subcommand names, without the `jupyter-` prefix.
Nested children (e.g. jupyter-sub-subsub) are not included.

### Function: _execvp(cmd, argv)

**Description:** execvp, except on Windows where it uses Popen

Python provides execvp on Windows, but its behavior is problematic (Python bug#9148).

### Function: _jupyter_abspath(subcommand)

**Description:** This method get the abspath of a specified jupyter-subcommand with no
changes on ENV.

### Function: _path_with_self()

**Description:** Put `jupyter`'s dir at the front of PATH

Ensures that /path/to/jupyter subcommand
will do /path/to/jupyter-subcommand
even if /other/jupyter-subcommand is ahead of it on PATH

### Function: _evaluate_argcomplete(parser)

**Description:** If argcomplete is enabled, trigger autocomplete or return current words

If the first word looks like a subcommand, return the current command
that is attempting to be completed so that the subcommand can evaluate it;
otherwise auto-complete using the main parser.

### Function: main()

**Description:** The command entry point.

### Function: epilog(self)

**Description:** Add subcommands to epilog on request

Avoids searching PATH for subcommands unless help output is requested.

### Function: epilog(self, x)

**Description:** Ignore epilog set in Parser.__init__

### Function: argcomplete(self)

**Description:** Trigger auto-completion, if enabled
