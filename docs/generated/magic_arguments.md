## AI Summary

A file named magic_arguments.py.


## Class: MagicHelpFormatter

**Description:** A HelpFormatter with a couple of changes to meet our needs.
    

## Class: MagicArgumentParser

**Description:** An ArgumentParser tweaked for use by IPython magics.
    

### Function: construct_parser(magic_func)

**Description:** Construct an argument parser using the function decorations.
    

### Function: parse_argstring(magic_func, argstring)

**Description:** Parse the string of arguments for the given magic function.
    

### Function: real_name(magic_func)

**Description:** Find the real name of the magic.
    

## Class: ArgDecorator

**Description:** Base class for decorators to add ArgumentParser information to a method.
    

## Class: magic_arguments

**Description:** Mark the magic as having argparse arguments and possibly adjust the
name.

## Class: ArgMethodWrapper

**Description:** Base class to define a wrapper for ArgumentParser method.

Child class must define either `_method_name` or `add_to_parser`.

## Class: argument

**Description:** Store arguments and keywords to pass to add_argument().

Instances also serve to decorate command methods.

## Class: defaults

**Description:** Store arguments and keywords to pass to set_defaults().

Instances also serve to decorate command methods.

## Class: argument_group

**Description:** Store arguments and keywords to pass to add_argument_group().

Instances also serve to decorate command methods.

## Class: kwds

**Description:** Provide other keywords to the sub-parser constructor.
    

### Function: _fill_text(self, text, width, indent)

### Function: _format_action_invocation(self, action)

### Function: add_usage(self, usage, actions, groups, prefix)

### Function: __init__(self, prog, usage, description, epilog, parents, formatter_class, prefix_chars, argument_default, conflict_handler, add_help)

### Function: error(self, message)

**Description:** Raise a catchable error instead of exiting.
        

### Function: parse_argstring(self, argstring)

**Description:** Split a string into an argument list and parse that argument list.
        

### Function: __call__(self, func)

### Function: add_to_parser(self, parser, group)

**Description:** Add this object's information to the parser, if necessary.
        

### Function: __init__(self, name)

### Function: __call__(self, func)

### Function: __init__(self)

### Function: add_to_parser(self, parser, group)

**Description:** Add this object's information to the parser.
        

### Function: add_to_parser(self, parser, group)

**Description:** Add this object's information to the parser.
        

### Function: __init__(self)

### Function: __call__(self, func)
