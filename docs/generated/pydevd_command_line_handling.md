## AI Summary

A file named pydevd_command_line_handling.py.


## Class: ArgHandlerWithParam

**Description:** Handler for some arguments which needs a value

## Class: ArgHandlerBool

**Description:** If a given flag is received, mark it as 'True' in setup.

### Function: convert_ppid(ppid)

### Function: get_pydevd_file()

### Function: setup_to_argv(setup, skip_names)

**Description:** :param dict setup:
    A dict previously gotten from process_command_line.

:param set skip_names:
    The names in the setup which shouldn't be converted to argv.

:note: does not handle --file nor --DEBUG.

### Function: process_command_line(argv)

**Description:** parses the arguments.
removes our arguments from the command line

### Function: __init__(self, arg_name, convert_val, default_val)

### Function: to_argv(self, lst, setup)

### Function: handle_argv(self, argv, i, setup)

### Function: __init__(self, arg_name, default_val)

### Function: to_argv(self, lst, setup)

### Function: handle_argv(self, argv, i, setup)
