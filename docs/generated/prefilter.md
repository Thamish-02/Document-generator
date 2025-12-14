## AI Summary

A file named prefilter.py.


## Class: PrefilterError

### Function: is_shadowed(identifier, ip)

**Description:** Is the given identifier defined in one of the namespaces which shadow
the alias and magic namespaces?  Note that an identifier is different
than ifun, because it can not contain a '.' character.

## Class: PrefilterManager

**Description:** Main prefilter component.

The IPython prefilter is run on all user input before it is run.  The
prefilter consumes lines of input and produces transformed lines of
input.

The implementation consists of two phases:

1. Transformers
2. Checkers and handlers

Over time, we plan on deprecating the checkers and handlers and doing
everything in the transformers.

The transformers are instances of :class:`PrefilterTransformer` and have
a single method :meth:`transform` that takes a line and returns a
transformed line.  The transformation can be accomplished using any
tool, but our current ones use regular expressions for speed.

After all the transformers have been run, the line is fed to the checkers,
which are instances of :class:`PrefilterChecker`.  The line is passed to
the :meth:`check` method, which either returns `None` or a
:class:`PrefilterHandler` instance.  If `None` is returned, the other
checkers are tried.  If an :class:`PrefilterHandler` instance is returned,
the line is passed to the :meth:`handle` method of the returned
handler and no further checkers are tried.

Both transformers and checkers have a `priority` attribute, that determines
the order in which they are called.  Smaller priorities are tried first.

Both transformers and checkers also have `enabled` attribute, which is
a boolean that determines if the instance is used.

Users or developers can change the priority or enabled attribute of
transformers or checkers, but they must call the :meth:`sort_checkers`
or :meth:`sort_transformers` method after changing the priority.

## Class: PrefilterTransformer

**Description:** Transform a line of user input.

## Class: PrefilterChecker

**Description:** Inspect an input line and return a handler for that line.

## Class: EmacsChecker

## Class: MacroChecker

## Class: IPyAutocallChecker

## Class: AssignmentChecker

## Class: AutoMagicChecker

## Class: PythonOpsChecker

## Class: AutocallChecker

## Class: PrefilterHandler

## Class: MacroHandler

## Class: MagicHandler

## Class: AutoHandler

## Class: EmacsHandler

### Function: __init__(self, shell)

### Function: sort_transformers(self)

**Description:** Sort the transformers by priority.

This must be called after the priority of a transformer is changed.
The :meth:`register_transformer` method calls this automatically.

### Function: transformers(self)

**Description:** Return a list of checkers, sorted by priority.

### Function: register_transformer(self, transformer)

**Description:** Register a transformer instance.

### Function: unregister_transformer(self, transformer)

**Description:** Unregister a transformer instance.

### Function: init_checkers(self)

**Description:** Create the default checkers.

### Function: sort_checkers(self)

**Description:** Sort the checkers by priority.

This must be called after the priority of a checker is changed.
The :meth:`register_checker` method calls this automatically.

### Function: checkers(self)

**Description:** Return a list of checkers, sorted by priority.

### Function: register_checker(self, checker)

**Description:** Register a checker instance.

### Function: unregister_checker(self, checker)

**Description:** Unregister a checker instance.

### Function: init_handlers(self)

**Description:** Create the default handlers.

### Function: handlers(self)

**Description:** Return a dict of all the handlers.

### Function: register_handler(self, name, handler, esc_strings)

**Description:** Register a handler instance by name with esc_strings.

### Function: unregister_handler(self, name, handler, esc_strings)

**Description:** Unregister a handler instance by name with esc_strings.

### Function: get_handler_by_name(self, name)

**Description:** Get a handler by its name.

### Function: get_handler_by_esc(self, esc_str)

**Description:** Get a handler by its escape string.

### Function: prefilter_line_info(self, line_info)

**Description:** Prefilter a line that has been converted to a LineInfo object.

This implements the checker/handler part of the prefilter pipe.

### Function: find_handler(self, line_info)

**Description:** Find a handler for the line_info by trying checkers.

### Function: transform_line(self, line, continue_prompt)

**Description:** Calls the enabled transformers in order of increasing priority.

### Function: prefilter_line(self, line, continue_prompt)

**Description:** Prefilter a single input line as text.

This method prefilters a single line of text by calling the
transformers and then the checkers/handlers.

### Function: prefilter_lines(self, lines, continue_prompt)

**Description:** Prefilter multiple input lines of text.

This is the main entry point for prefiltering multiple lines of
input.  This simply calls :meth:`prefilter_line` for each line of
input.

This covers cases where there are multiple lines in the user entry,
which is the case when the user goes back to a multiline history
entry and presses enter.

### Function: __init__(self, shell, prefilter_manager)

### Function: transform(self, line, continue_prompt)

**Description:** Transform a line, returning the new one.

### Function: __repr__(self)

### Function: __init__(self, shell, prefilter_manager)

### Function: check(self, line_info)

**Description:** Inspect line_info and return a handler instance or None.

### Function: __repr__(self)

### Function: check(self, line_info)

**Description:** Emacs ipython-mode tags certain input lines.

### Function: check(self, line_info)

### Function: check(self, line_info)

**Description:** Instances of IPyAutocall in user_ns get autocalled immediately

### Function: check(self, line_info)

**Description:** Check to see if user is assigning to a var for the first time, in
which case we want to avoid any sort of automagic / autocall games.

This allows users to assign to either alias or magic names true python
variables (the magic/alias systems always take second seat to true
python code).  E.g. ls='hi', or ls,that=1,2

### Function: check(self, line_info)

**Description:** If the ifun is magic, and automagic is on, run it.  Note: normal,
non-auto magic would already have been triggered via '%' in
check_esc_chars. This just checks for automagic.  Also, before
triggering the magic handler, make sure that there is nothing in the
user namespace which could shadow it.

### Function: check(self, line_info)

**Description:** If the 'rest' of the line begins with a function call or pretty much
any python operator, we should simply execute the line (regardless of
whether or not there's a possible autocall expansion).  This avoids
spurious (and very confusing) geattr() accesses.

### Function: check(self, line_info)

**Description:** Check if the initial word/function is callable and autocall is on.

### Function: __init__(self, shell, prefilter_manager)

### Function: handle(self, line_info)

**Description:** Handle normal input lines. Use as a template for handlers.

### Function: __str__(self)

### Function: handle(self, line_info)

### Function: handle(self, line_info)

**Description:** Execute magic functions.

### Function: handle(self, line_info)

**Description:** Handle lines which can be auto-executed, quoting if requested.

### Function: handle(self, line_info)

**Description:** Handle input lines marked by python-mode.
